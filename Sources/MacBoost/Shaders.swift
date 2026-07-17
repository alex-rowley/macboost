// MSL compute kernels for MacBoost, compiled at runtime.
//
// Data layout notes:
//  - Binned feature matrix is TILED uint8: features are grouped 4 per
//    32-bit word, laid out bins[tile * numSamples * 4 + i * 4 + lane] with
//    tile = f / 4, lane = f % 4. One aligned uint32 load fetches 4 features
//    of one sample; the histogram kernel builds 4 feature histograms per
//    threadgroup pass so the per-sample order/gradient reads are amortised
//    over 4 features instead of re-read per feature.
//  - Bin layout: bins 0..numBins-2 hold data (quantile bins for numeric
//    features, category ids for categorical ones); the LAST bin
//    (numBins - 1) is reserved for missing values (NaN).
//  - Gradients are quantised per tree: grad to signed 16-bit against the
//    tree's max |grad| (GPU-reduced), hess to 7-bit against the objective's
//    analytic hess bound. Packed per sample as (qg << 16) | qh. Threadgroup
//    histogram accumulation is then 2 native integer atomics per
//    (sample, feature): grad into atomic_int, (qh << 13 | 1) into
//    atomic_uint so hess and count share one word. Overflow bounds at
//    samplesPerGroup = 4096: count <= 4096 < 2^13, hess <= 4096*127 < 2^19,
//    grad <= 4096*32767 < 2^31. Device histograms stay float (dequantised
//    on flush), so split search and host code are precision-agnostic.
//  - Histograms are [node][feature][bin][3] float32: (grad, hess, count).
//  - Tree nodes use implicit heap indexing: children of n are 2n+1 / 2n+2.
//  - NodeSplit.flags: bit0 = missing values go left, bit1 = categorical
//    split (left iff catMask bit for the sample's bin is set).

let mslSource = """
#include <metal_stdlib>
using namespace metal;

#define HIST_CH 3
#define TG_SIZE 256
#define TILE_F 8
#define FLAG_DEFAULT_LEFT 1u
#define FLAG_CATEGORICAL  2u

inline ulong bin_index(uint f, uint i, uint numSamples) {
    return (ulong)(f >> 3) * numSamples * TILE_F + (ulong)i * TILE_F + (f & 7u);
}

struct BinParams { uint numSamples; uint numFeatures; uint numBins; };

// One thread per (sample, feature). Numeric features binary-search the
// quantile edges; categorical features map the (validated) category id
// straight to a bin; NaN goes to the reserved missing bin.
kernel void bin_data(
    device const float *X         [[buffer(0)]],   // feature-major raw values
    device const float *edges     [[buffer(1)]],   // f * (numBins-2) upper edges
    device uchar       *bins      [[buffer(2)]],   // tiled layout (see header)
    device const uchar *featFlags [[buffer(3)]],   // 1 = categorical
    constant BinParams &p         [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x, f = gid.y;
    if (i >= p.numSamples || f >= p.numFeatures) return;
    uint dataBins = p.numBins - 1;
    float v = X[(ulong)f * p.numSamples + i];
    uchar b;
    if (isnan(v)) {
        b = uchar(dataBins);                        // missing bin
    } else if (featFlags[f] == 1) {
        b = uchar(clamp(int(rint(v)), 0, int(dataBins) - 1));
    } else {
        device const float *e = edges + (ulong)f * (dataBins - 1);
        uint lo = 0, hi = dataBins - 1;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1;
            if (v > e[mid]) lo = mid + 1; else hi = mid;
        }
        b = uchar(lo);
    }
    bins[bin_index(f, i, p.numSamples)] = b;
}

// Objectives: 0=l2 1=binary logistic (aux = scale_pos_weight) 2=mae
// 3=huber (alpha = delta) 4=quantile (alpha) 5=poisson 6=tweedie (aux =
// variance power). Emits per-sample (g, h) with optional sample weights
// applied, plus simdgroup-reduced device maxima of |g| and h that the
// quantisation and histogram kernels read in the same command buffer
// (dynamic hessian scaling is what lets Poisson/Tweedie/weighted hessians
// keep full 7-bit precision).
struct GradParams { uint numSamples; uint objective; float alpha; float aux; uint hasWeights; };

inline void reduce_max_gh(float g, float h, device atomic_uint *maxGH) {
    float mg = simd_max(fabs(g));
    float mh = simd_max(h);
    if (simd_is_first()) {
        atomic_fetch_max_explicit(&maxGH[0], as_type<uint>(mg), memory_order_relaxed);
        atomic_fetch_max_explicit(&maxGH[1], as_type<uint>(mh), memory_order_relaxed);
    }
}

kernel void compute_gradients(
    device const float *preds   [[buffer(0)]],
    device const float *labels  [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device float2      *gh      [[buffer(3)]],
    device atomic_uint *maxGH   [[buffer(4)]],   // [0]=max|g|, [1]=max h (f32 bits)
    constant GradParams &p      [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    float g = 0.0f, h = 0.0f;
    if (i < p.numSamples) {
        float f = preds[i], y = labels[i];
        switch (p.objective) {
        case 1: {   // binary logistic
            float q = 1.0f / (1.0f + exp(-f));
            float w = y > 0.5f ? p.aux : 1.0f;    // scale_pos_weight
            g = (q - y) * w; h = max(q * (1.0f - q), 1e-6f) * w; break;
        }
        case 2:     // mae
            g = f > y ? 1.0f : -1.0f; h = 1.0f; break;
        case 3: {   // huber (delta = alpha)
            g = clamp(f - y, -p.alpha, p.alpha); h = 1.0f; break;
        }
        case 4:     // quantile (pinball)
            g = f > y ? (1.0f - p.alpha) : -p.alpha; h = 1.0f; break;
        case 5: {   // poisson, log link
            float mu = exp(clamp(f, -30.0f, 30.0f));
            g = mu - y; h = max(mu, 1e-6f); break;
        }
        case 6: {   // tweedie (variance power = aux), log link
            float fc = clamp(f, -30.0f, 30.0f);
            float e2 = exp((2.0f - p.aux) * fc), e1 = exp((1.0f - p.aux) * fc);
            g = e2 - y * e1;
            h = max((2.0f - p.aux) * e2 - (1.0f - p.aux) * y * e1, 1e-6f); break;
        }
        default:    // l2
            g = f - y; h = 1.0f; break;
        }
        if (p.hasWeights != 0) { float w = weights[i]; g *= w; h *= w; }
        gh[i] = float2(g, h);
    }
    reduce_max_gh(g, h, maxGH);
}

struct GradMCParams { uint numSamples; uint numClasses; uint classIndex; uint hasWeights; };

// Softmax multiclass gradients for one class's tree. preds is class-major
// (k * numSamples + i). Sequential within-round updates (class k sees the
// effect of classes < k's trees this round), like XGBoost's default.
kernel void compute_gradients_multiclass(
    device const float *preds   [[buffer(0)]],
    device const float *labels  [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device float2      *gh      [[buffer(3)]],
    device atomic_uint *maxGH   [[buffer(4)]],
    constant GradMCParams &p    [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    float g = 0.0f, h = 0.0f;
    if (i < p.numSamples) {
        float m = -INFINITY;
        for (uint k = 0; k < p.numClasses; ++k)
            m = max(m, preds[k * p.numSamples + i]);
        float sum = 0.0f;
        for (uint k = 0; k < p.numClasses; ++k)
            sum += exp(preds[k * p.numSamples + i] - m);
        float pk = exp(preds[p.classIndex * p.numSamples + i] - m) / sum;
        float yk = uint(labels[i]) == p.classIndex ? 1.0f : 0.0f;
        g = pk - yk;
        h = max(2.0f * pk * (1.0f - pk), 1e-6f);
        if (p.hasWeights != 0) { float w = weights[i]; g *= w; h *= w; }
        gh[i] = float2(g, h);
    }
    reduce_max_gh(g, h, maxGH);
}

struct QuantParams { uint numSamples; };

// Pack per-sample gradients: signed 16-bit grad in the high half, 7-bit
// hess in the low half, both scaled against the GPU-reduced maxima.
kernel void quantize_gradients(
    device const float2 *gh    [[buffer(0)]],
    device const uint   *maxGH [[buffer(1)]],
    device uint         *ghq   [[buffer(2)]],
    constant QuantParams &p    [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.numSamples) return;
    float mg = as_type<float>(maxGH[0]);
    float mh = as_type<float>(maxGH[1]);
    float gscale = mg > 0.0f ? 32767.0f / mg : 0.0f;
    float hscale = mh > 0.0f ? 127.0f / mh : 0.0f;
    float2 v = gh[i];
    int  qg = int(rint(v.x * gscale));
    uint qh = uint(rint(v.y * hscale));
    ghq[i] = (uint(qg & 0xFFFF) << 16) | (qh & 0x7Fu);
}

// One-time identity permutation: level 0's "order" is all samples.
kernel void iota(
    device uint   *order [[buffer(0)]],
    constant uint &n     [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) order[i] = i;
}

struct HistParams {
    uint numSamples; uint numFeatures; uint numBins;
    uint numNodes; uint samplesPerGroup;
};

// One threadgroup per (sample-block, feature-tile, node). Reads each
// sample's order index, packed gradient and 4-feature bin word ONCE, then
// accumulates 4 feature histograms in threadgroup memory with native
// integer atomics. Flush dequantises into the float device histograms.
kernel void build_histograms(
    device const uchar  *binsRaw   [[buffer(0)]],
    device const uint   *ghq       [[buffer(1)]],
    device const uint   *order     [[buffer(2)]],
    device const uint   *nodeStart [[buffer(3)]],
    device const uint   *nodeCount [[buffer(4)]],
    device atomic_float *hist      [[buffer(5)]],
    device const uint   *maxGH     [[buffer(6)]],
    constant HistParams &p         [[buffer(7)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
    uint node = tg.z, tile = tg.y;
    uint count = nodeCount[node];
    uint blockStart = tg.x * p.samplesPerGroup;
    if (blockStart >= count) return;   // uniform per threadgroup: safe early-out

    threadgroup atomic_int  lg[TILE_F * 256];    // grad sums
    threadgroup atomic_uint lhc[TILE_F * 256];   // (hess << 13) | count
    for (uint b = tid; b < TILE_F * 256; b += TG_SIZE) {
        atomic_store_explicit(&lg[b], 0, memory_order_relaxed);
        atomic_store_explicit(&lhc[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint blockEnd = min(blockStart + p.samplesPerGroup, count);
    device const uint2 *tiles = (device const uint2 *)binsRaw + (ulong)tile * p.numSamples;
    device const uint *ord = order + nodeStart[node];
    for (uint s = blockStart + tid; s < blockEnd; s += TG_SIZE) {
        uint i = ord[s];
        uint q = ghq[i];
        int  qg = int(short(ushort(q >> 16)));
        uint hc = ((q & 0x7Fu) << 13) | 1u;
        uint2 bins8 = tiles[i];
        for (uint lane = 0; lane < TILE_F; ++lane) {
            uint word = lane < 4 ? bins8.x : bins8.y;
            uint b = (word >> (8u * (lane & 3u))) & 0xFFu;
            atomic_fetch_add_explicit(&lg[lane * 256 + b], qg, memory_order_relaxed);
            atomic_fetch_add_explicit(&lhc[lane * 256 + b], hc, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gInvScale = as_type<float>(maxGH[0]) / 32767.0f;
    float hInvScale = as_type<float>(maxGH[1]) / 127.0f;
    // Occupancy-proportional path: when ONE threadgroup owns the node
    // (count <= samplesPerGroup — every node once rows-per-node is small),
    // write the slice densely instead of accumulating, so it needs no
    // zeroing pass at all. 0 + x == x in fp, so values are bit-identical
    // to the zero+add path. Multi-group nodes keep atomic accumulation
    // into a slice that zero_built pre-zeroed.
    bool single = count <= p.samplesPerGroup;
    for (uint idx = tid; idx < TILE_F * 256; idx += TG_SIZE) {
        uint lane = idx >> 8, b = idx & 0xFFu;
        uint f = tile * TILE_F + lane;
        if (f >= p.numFeatures || b >= p.numBins) continue;
        int  gsum = atomic_load_explicit(&lg[idx], memory_order_relaxed);
        uint hc   = atomic_load_explicit(&lhc[idx], memory_order_relaxed);
        ulong off = (((ulong)node * p.numFeatures + f) * p.numBins + b) * HIST_CH;
        if (single) {
            device float *hw = (device float *)hist + off;
            hw[0] = float(gsum) * gInvScale;
            hw[1] = float(hc >> 13) * hInvScale;
            hw[2] = float(hc & 0x1FFFu);
        } else {
            if (gsum == 0 && hc == 0u) continue;
            device atomic_float *h = hist + off;
            atomic_fetch_add_explicit(&h[0], float(gsum) * gInvScale, memory_order_relaxed);
            atomic_fetch_add_explicit(&h[1], float(hc >> 13) * hInvScale, memory_order_relaxed);
            atomic_fetch_add_explicit(&h[2], float(hc & 0x1FFFu), memory_order_relaxed);
        }
    }
}

struct ZeroBuiltParams { uint sliceLen; uint numNodes; uint samplesPerGroup; };

// Zero only the histogram slices that will be accumulated into by more
// than one threadgroup. Single-group builds dense-write their slice,
// derived siblings are dense-written by subtraction, and dead nodes are
// never read (find_splits' liveness guard) — none of them need zeroing.
kernel void zero_built(
    device const uint     *nodeCount [[buffer(0)]],
    device float          *hist      [[buffer(1)]],
    constant ZeroBuiltParams &p      [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint j = gid.x, node = gid.y;
    if (j >= p.sliceLen || node >= p.numNodes) return;
    uint c = nodeCount[node];
    if (c == 0 || c <= p.samplesPerGroup) return;
    hist[(ulong)node * p.sliceLen + j] = 0.0f;
}

struct SubParams { uint sliceLen; };   // sliceLen = F * bins * 3

// LightGBM's histogram-subtraction trick: the larger child's histogram is
// parent - smallerChild, computed bin-wise. `map` holds (derive, parent,
// sibling) local-node index triples; the live count is GPU-written by
// decide_splits, so the grid over-dispatches and early-outs.
kernel void subtract_histograms(
    device const float *prev      [[buffer(0)]],   // previous level's histograms
    device float       *cur       [[buffer(1)]],
    device const uint  *map       [[buffer(2)]],
    device const uint  *numDerive [[buffer(3)]],
    constant SubParams &p         [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint j = gid.x;
    if (j >= p.sliceLen || gid.y >= numDerive[0]) return;
    uint d = map[gid.y * 3], par = map[gid.y * 3 + 1], sib = map[gid.y * 3 + 2];
    cur[(ulong)d * p.sliceLen + j] =
        prev[(ulong)par * p.sliceLen + j] - cur[(ulong)sib * p.sliceLen + j];
}

struct SplitParams {
    uint numFeatures; uint numBins; uint numNodes;
    float lambda; float minChildHess; float minSplitGain; float catSmooth;
    uint levelStart;
};

struct SplitResult { float gain; uint bin; float gl; float hl; uint flags; };

// Stable ascending insertion sort of non-empty bins by g/(h + smooth) —
// LightGBM's Fisher ordering. Shared by find_splits (to score subsets) and
// decide_splits (to reconstruct the winning subset's bitmask).
inline int cat_sort(device const float *h, uint numBins, float catSmooth,
                    thread ushort *order_, thread float *key)
{
    int m = 0;
    for (uint b = 0; b < numBins; ++b) {
        float hb = h[b*HIST_CH+1];
        if (hb > 0.0f) {
            order_[m] = ushort(b);
            key[m] = h[b*HIST_CH] / (hb + catSmooth);
            ++m;
        }
    }
    for (int i = 1; i < m; ++i) {
        ushort ob = order_[i]; float ok = key[i]; int j = i - 1;
        while (j >= 0 && key[j] > ok) {
            key[j+1] = key[j]; order_[j+1] = order_[j]; --j;
        }
        key[j+1] = ok; order_[j+1] = ob;
    }
    return m;
}

// One thread per (feature, node), XGBoost-style gain with L2 regularisation.
// Numeric features: prefix scan over data bins, trying the missing bin on
// each side (learned default direction). Categorical features: LightGBM's
// Fisher method — sort non-empty bins by grad/hess ratio, scan prefixes of
// that order; `bin` in the result is the left-subset size k. gl/hl are
// always the ACTUAL left-child totals (missing/subset included).
kernel void find_splits(
    device const float  *hist      [[buffer(0)]],
    device const uchar  *featFlags [[buffer(1)]],
    device const uchar  *featMask  [[buffer(2)]],   // 0 = excluded this tree
    device const char   *monotone  [[buffer(3)]],   // per feature: -1/0/+1
    device SplitResult  *results   [[buffer(4)]],
    device const float4 *stats     [[buffer(5)]],
    constant SplitParams &p        [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint f = gid.x, node = gid.y;
    if (f >= p.numFeatures || node >= p.numNodes) return;
    // Liveness guard: dead nodes' histogram slices are never zeroed nor
    // written, so never read them. stats.z (sample count) is written by
    // decide_splits for both children of every real split; the root is
    // always live.
    if (p.levelStart > 0 && stats[p.levelStart + node].z < 1.0f) {
        SplitResult none;
        none.gain = -INFINITY; none.bin = 0; none.gl = 0; none.hl = 0; none.flags = 0;
        results[node * p.numFeatures + f] = none;
        return;
    }
    if (featMask[f] == 0) {
        SplitResult none;
        none.gain = -INFINITY; none.bin = 0; none.gl = 0; none.hl = 0; none.flags = 0;
        results[node * p.numFeatures + f] = none;
        return;
    }
    device const float *h =
        hist + (((ulong)node * p.numFeatures + f) * p.numBins) * HIST_CH;
    uint dataBins = p.numBins - 1;

    float G = 0.0f, H = 0.0f;
    for (uint b = 0; b < p.numBins; ++b) { G += h[b*HIST_CH]; H += h[b*HIST_CH+1]; }
    float parentScore = (G * G) / (H + p.lambda);

    float bestGain = -INFINITY; uint bestBin = 0;
    float bgl = 0.0f, bhl = 0.0f; uint bflags = 0;

    if (featFlags[f] == 1) {
        // Categorical: the missing bin participates as one more category.
        ushort order_[256]; float key[256];
        int m = cat_sort(h, p.numBins, p.catSmooth, order_, key);
        float gl = 0.0f, hl = 0.0f;
        for (int i = 0; i + 1 < m; ++i) {
            uint b = order_[i];
            gl += h[b*HIST_CH]; hl += h[b*HIST_CH+1];
            float gr = G - gl, hr = H - hl;
            if (hl < p.minChildHess || hr < p.minChildHess) continue;
            float gain = gl*gl/(hl+p.lambda) + gr*gr/(hr+p.lambda) - parentScore;
            if (gain > bestGain) {
                bestGain = gain; bestBin = uint(i + 1);
                bgl = gl; bhl = hl; bflags = FLAG_CATEGORICAL;
            }
        }
    } else {
        float Gm = h[dataBins*HIST_CH], Hm = h[dataBins*HIST_CH+1];
        int cst = int(monotone[f]);
        float gl = 0.0f, hl = 0.0f;
        for (uint b = 0; b + 1 < dataBins; ++b) {
            gl += h[b*HIST_CH]; hl += h[b*HIST_CH+1];
            {   // missing goes right
                float rg = G - gl, rh = H - hl;
                if (hl >= p.minChildHess && rh >= p.minChildHess) {
                    bool ok = true;
                    if (cst != 0) {
                        float vl = -gl / (hl + p.lambda), vr = -rg / (rh + p.lambda);
                        ok = cst > 0 ? vl <= vr : vl >= vr;
                    }
                    float gain = gl*gl/(hl+p.lambda) + rg*rg/(rh+p.lambda) - parentScore;
                    if (ok && gain > bestGain) {
                        bestGain = gain; bestBin = b; bgl = gl; bhl = hl; bflags = 0;
                    }
                }
            }
            if (Hm > 0.0f) {   // missing goes left
                float lg = gl + Gm, lh = hl + Hm;
                float rg = G - lg, rh = H - lh;
                if (lh >= p.minChildHess && rh >= p.minChildHess) {
                    bool ok = true;
                    if (cst != 0) {
                        float vl = -lg / (lh + p.lambda), vr = -rg / (rh + p.lambda);
                        ok = cst > 0 ? vl <= vr : vl >= vr;
                    }
                    float gain = lg*lg/(lh+p.lambda) + rg*rg/(rh+p.lambda) - parentScore;
                    if (ok && gain > bestGain) {
                        bestGain = gain; bestBin = b; bgl = lg; bhl = lh;
                        bflags = FLAG_DEFAULT_LEFT;
                    }
                }
            }
        }
    }
    SplitResult r;
    r.gain = bestGain; r.bin = bestBin; r.gl = bgl; r.hl = bhl; r.flags = bflags;
    results[node * p.numFeatures + f] = r;
}

struct NodeSplit { int feature; uint bin; uint flags; };  // feature < 0 => leaf

inline bool goes_left(uchar b, NodeSplit s, uint numBins,
                      device const uint *catMask, uint node)
{
    if (s.flags & FLAG_CATEGORICAL)
        return (catMask[node * 8u + b / 32u] >> (b % 32u)) & 1u;
    if (b == uchar(numBins - 1))
        return (s.flags & FLAG_DEFAULT_LEFT) != 0;
    return b <= uchar(s.bin);
}

// --- Leaf-wise growth: single-segment stable partition -----------------
// Splitting one leaf partitions ITS slice of the order buffer into
// left|right. Three dispatches: per-group left counts, a serial exclusive
// scan, then a chunked stable scatter into a scratch buffer (each thread
// owns a contiguous chunk, so within-chunk order is preserved and chunk
// bases come from a threadgroup scan).

struct PartitionParams {
    uint segStart; uint segCount; uint numSamples; uint numBins;
    uint samplesPerGroup; uint feature; uint bin; uint flags;
};

inline bool part_left(uchar b, device const uint *mask, constant PartitionParams &p) {
    if (p.flags & FLAG_CATEGORICAL)
        return (mask[b / 32u] >> (b % 32u)) & 1u;
    if (b == uchar(p.numBins - 1))
        return (p.flags & FLAG_DEFAULT_LEFT) != 0;
    return b <= uchar(p.bin);
}

kernel void partition_count(
    device const uchar *bins      [[buffer(0)]],
    device const uint  *order     [[buffer(1)]],
    device const uint  *mask      [[buffer(2)]],   // 8 words (categorical)
    device uint        *groupLeft [[buffer(3)]],
    constant PartitionParams &p   [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint g = tg.x;
    uint gStart = g * p.samplesPerGroup;
    threadgroup atomic_uint tot;
    if (tid == 0) atomic_store_explicit(&tot, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gStart < p.segCount) {
        uint gEnd = min(gStart + p.samplesPerGroup, p.segCount);
        uint cnt = 0;
        for (uint sIdx = gStart + tid; sIdx < gEnd; sIdx += TG_SIZE) {
            uint i = order[p.segStart + sIdx];
            uchar b = bins[bin_index(p.feature, i, p.numSamples)];
            if (part_left(b, mask, p)) ++cnt;
        }
        atomic_fetch_add_explicit(&tot, cnt, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) groupLeft[g] = atomic_load_explicit(&tot, memory_order_relaxed);
}

kernel void partition_scan(
    device uint   *groupLeft [[buffer(0)]],   // in: counts, out: exclusive starts
    device uint   *totals    [[buffer(1)]],   // [0] = total lefts
    constant uint &numGroups [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0) return;
    uint acc = 0;
    for (uint g = 0; g < numGroups; ++g) {
        uint c = groupLeft[g];
        groupLeft[g] = acc;
        acc += c;
    }
    totals[0] = acc;
}

kernel void partition_scatter(
    device const uchar *bins      [[buffer(0)]],
    device const uint  *order     [[buffer(1)]],
    device const uint  *mask      [[buffer(2)]],
    device const uint  *groupLeft [[buffer(3)]],   // exclusive starts
    device const uint  *totals    [[buffer(4)]],
    device uint        *outOrder  [[buffer(5)]],   // scratch, same index space
    constant PartitionParams &p   [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint g = tg.x;
    uint gStart = g * p.samplesPerGroup;
    if (gStart >= p.segCount) return;
    uint gEnd = min(gStart + p.samplesPerGroup, p.segCount);
    uint n = gEnd - gStart;
    uint per = (n + TG_SIZE - 1) / TG_SIZE;
    uint cStart = gStart + min(tid * per, n);
    uint cEnd = gStart + min((tid + 1) * per, n);

    uint myLeft = 0;
    for (uint sIdx = cStart; sIdx < cEnd; ++sIdx) {
        uint i = order[p.segStart + sIdx];
        uchar b = bins[bin_index(p.feature, i, p.numSamples)];
        if (part_left(b, mask, p)) ++myLeft;
    }
    threadgroup uint lefts[TG_SIZE];
    threadgroup uint lbase[TG_SIZE], rbase[TG_SIZE];
    lefts[tid] = myLeft;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint la = 0, ra = 0;
        for (uint t = 0; t < TG_SIZE; ++t) {
            uint tn = min((t + 1) * per, n) - min(t * per, n);
            lbase[t] = la; rbase[t] = ra;
            la += lefts[t]; ra += tn - lefts[t];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint leftTotal = totals[0];
    uint lpos = p.segStart + groupLeft[g] + lbase[tid];
    uint rpos = p.segStart + leftTotal + (gStart - groupLeft[g]) + rbase[tid];
    for (uint sIdx = cStart; sIdx < cEnd; ++sIdx) {
        uint i = order[p.segStart + sIdx];
        uchar b = bins[bin_index(p.feature, i, p.numSamples)];
        if (part_left(b, mask, p)) outOrder[lpos++] = i;
        else                       outOrder[rpos++] = i;
    }
}

// sibling = parent - child, in place in the parent's histogram slot.
kernel void subtract_slot(
    device float       *parent [[buffer(0)]],
    device const float *child  [[buffer(1)]],
    constant uint      &len    [[buffer(2)]],
    uint j [[thread_position_in_grid]])
{
    if (j < len) parent[j] -= child[j];
}

// GOSS (gradient-based one-side sampling), LightGBM-style: keep the
// top-rate fraction of samples by |gradient| plus a uniform sample of the
// rest, amplifying the latter's gradients by (1-a)/b. The threshold is the
// approximate top-rate quantile from a 1024-bucket |g| histogram; kept
// indices are compacted into the level-0 order buffer.

#define GOSS_BUCKETS 1024

kernel void goss_grad_hist(
    device const float2 *gh      [[buffer(0)]],
    device const uint   *maxAbsG [[buffer(1)]],
    device atomic_uint  *buckets [[buffer(2)]],
    constant uint       &n       [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float mg = as_type<float>(maxAbsG[0]);
    if (mg <= 0.0f) return;
    uint b = min(uint(GOSS_BUCKETS - 1),
                 uint(fabs(gh[i].x) / mg * float(GOSS_BUCKETS)));
    atomic_fetch_add_explicit(&buckets[b], 1u, memory_order_relaxed);
}

// Scan buckets from the top until topCount samples are covered; the bucket's
// lower edge becomes the keep-threshold (ties above it are all kept).
kernel void goss_threshold(
    device const uint *buckets   [[buffer(0)]],
    device const uint *maxAbsG   [[buffer(1)]],
    device uint       *threshold [[buffer(2)]],   // f32 bits
    constant uint     &topCount  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    float mg = as_type<float>(maxAbsG[0]);
    uint cum = 0;
    uint b = GOSS_BUCKETS;
    while (b > 0 && cum < topCount) {
        b -= 1;
        cum += buckets[b];
    }
    float t = mg > 0.0f ? float(b) / float(GOSS_BUCKETS) * mg : 0.0f;
    threshold[0] = as_type<uint>(t);
}

struct GossParams {
    uint numSamples; uint treeSeed; float otherProb; float weight;
};

inline uint wang_hash(uint x) {
    x = (x ^ 61u) ^ (x >> 16); x *= 9u; x ^= x >> 4;
    x *= 0x27d4eb2du; x ^= x >> 15;
    return x;
}

// Keep top-gradient samples (weight 1) and a hashed uniform sample of the
// rest (weight (1-a)/b, applied to grad AND hess in place). Kept indices
// are appended to orderOut via threadgroup-aggregated reservation; the
// weighted max |grad| feeds quantisation.
kernel void goss_select(
    device float2      *gh        [[buffer(0)]],
    device const uint  *threshold [[buffer(1)]],
    device uint        *orderOut  [[buffer(2)]],
    device atomic_uint *cursor    [[buffer(3)]],
    device atomic_uint *maxWGH    [[buffer(4)]],   // weighted max |g|, max h (f32 bits)
    constant GossParams &p        [[buffer(5)]],
    uint i   [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint localCnt;
    threadgroup uint blockBase;
    if (tid == 0) atomic_store_explicit(&localCnt, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool keep = false;
    float w = 1.0f;
    uint rank = 0;
    float g = 0.0f, hh = 0.0f;
    if (i < p.numSamples) {
        float2 v = gh[i];
        g = fabs(v.x);
        float t = as_type<float>(threshold[0]);
        if (g >= t) {
            keep = true;
        } else {
            uint h = wang_hash(i ^ (p.treeSeed * 0x9E3779B9u));
            if (float(h >> 8) * (1.0f / 16777216.0f) < p.otherProb) {
                keep = true; w = p.weight;
            }
        }
        hh = v.y * w;
        if (keep && w != 1.0f) gh[i] = float2(v.x * w, hh);
    }
    if (keep) rank = atomic_fetch_add_explicit(&localCnt, 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint cnt = atomic_load_explicit(&localCnt, memory_order_relaxed);
        blockBase = cnt > 0
            ? atomic_fetch_add_explicit(cursor, cnt, memory_order_relaxed) : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (keep) {
        orderOut[blockBase + rank] = i;
        atomic_fetch_max_explicit(&maxWGH[0], as_type<uint>(g * w), memory_order_relaxed);
        atomic_fetch_max_explicit(&maxWGH[1], as_type<uint>(hh), memory_order_relaxed);
    }
}

struct GossFinalizeParams { uint samplesPerGroup; uint numTiles; };

kernel void goss_finalize(
    device const uint *cursor       [[buffer(0)]],
    device uint       *buildCount0  [[buffer(1)]],
    device uint       *activeTotal0 [[buffer(2)]],
    device uint       *histArgs0    [[buffer(3)]],
    device uint       *routeArgs0   [[buffer(4)]],
    constant GossFinalizeParams &p  [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    uint kept = cursor[0];
    buildCount0[0] = kept;
    activeTotal0[0] = kept;
    histArgs0[0] = (kept + p.samplesPerGroup - 1) / p.samplesPerGroup;
    histArgs0[1] = p.numTiles;
    histArgs0[2] = 1;
    routeArgs0[0] = (kept + TG_SIZE - 1) / TG_SIZE;
    routeArgs0[1] = 1;
    routeArgs0[2] = 1;
}

struct RouteParams {
    uint levelStart; uint numLevel; uint numSamples; uint numBins;
    uint terminal; uint apply; uint predOffset;
};

// The per-level sample pass: walks the CURRENT level's node-grouped order
// entries (work is proportional to LIVE samples, shrinking as nodes die).
// Samples in leaf nodes get the leaf value applied to their prediction here
// and leave the active set; samples in split nodes are routed into their
// child's segment of the next order buffer. With `terminal` set (deepest
// level) children are leaves, so their values apply immediately instead of
// forwarding.
// Forwarding uses three-phase threadgroup aggregation to avoid hammering a
// handful of device cursors with millions of atomics at shallow levels:
// rank within the threadgroup via fast local atomics, reserve one block per
// (threadgroup, child) with a single device atomic, then write base + rank.
#define ROUTE_MAX_LOCAL 1024

kernel void route_samples(
    device const uint      *orderIn     [[buffer(0)]],
    device uint            *orderOut    [[buffer(1)]],
    device const uint      *segStart    [[buffer(2)]],   // this level, monotone
    device atomic_uint     *cursorsNext [[buffer(3)]],
    device const uint      *activeTotal [[buffer(4)]],
    device const NodeSplit *splits      [[buffer(5)]],
    device const float     *leafValues  [[buffer(6)]],
    device const uint      *catMask     [[buffer(7)]],
    device const uchar     *bins        [[buffer(8)]],
    device float           *preds       [[buffer(9)]],
    constant RouteParams   &p           [[buffer(10)]],
    uint s   [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint localCnt[ROUTE_MAX_LOCAL];
    threadgroup uint blockBase[ROUTE_MAX_LOCAL];
    uint numNext = p.numLevel * 2;
    bool agg = p.terminal == 0 && numNext <= ROUTE_MAX_LOCAL;
    if (agg) {
        for (uint c = tid; c < numNext; c += TG_SIZE)
            atomic_store_explicit(&localCnt[c], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // No early returns: every thread reaches the barriers below.
    uint i = 0, cl = 0, rank = 0;
    bool forwards = false;
    if (s < activeTotal[0]) {
        // Owning node: largest local index with segStart <= s. Dead nodes
        // carry duplicated starts; the later (live) index wins.
        uint lo = 0, hi = p.numLevel - 1;
        while (lo < hi) {
            uint mid = (lo + hi + 1) >> 1;
            if (segStart[mid] <= s) lo = mid; else hi = mid - 1;
        }
        uint pAbs = p.levelStart + lo;
        i = orderIn[s];
        NodeSplit sp = splits[pAbs];
        if (sp.feature < 0) {                 // leaf: settle this sample now
            if (p.apply) preds[p.predOffset + i] += leafValues[pAbs];
        } else {
            uchar b = bins[bin_index(uint(sp.feature), i, p.numSamples)];
            uint child = 2 * pAbs + 1
                       + (goes_left(b, sp, p.numBins, catMask, pAbs) ? 0 : 1);
            if (p.terminal) {                 // children are final leaves
                if (p.apply) preds[p.predOffset + i] += leafValues[child];
            } else {
                forwards = true;
                cl = child - (2 * p.levelStart + 1);
            }
        }
    }

    if (agg) {
        if (forwards)
            rank = atomic_fetch_add_explicit(&localCnt[cl], 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = tid; c < numNext; c += TG_SIZE) {
            uint cnt = atomic_load_explicit(&localCnt[c], memory_order_relaxed);
            blockBase[c] = cnt > 0
                ? atomic_fetch_add_explicit(&cursorsNext[c], cnt, memory_order_relaxed)
                : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (forwards) orderOut[blockBase[cl] + rank] = i;
    } else if (forwards) {
        uint pos = atomic_fetch_add_explicit(&cursorsNext[cl], 1u, memory_order_relaxed);
        orderOut[pos] = i;
    }
}

struct InitParams { uint rows; uint samplesPerGroup; uint numTiles; };

// Per-tree GPU-side reset of level 0's metadata (one root segment holding
// every sample) and its dispatch arguments.
kernel void init_tree(
    device uint   *segStart0    [[buffer(0)]],
    device uint   *buildCount0  [[buffer(1)]],
    device uint   *activeTotal0 [[buffer(2)]],
    device uint   *numDerive    [[buffer(3)]],
    device uint   *histArgs0    [[buffer(4)]],
    device uint   *routeArgs0   [[buffer(5)]],
    device float2 *bounds       [[buffer(6)]],   // leaf-value bounds (monotone)
    constant InitParams &p      [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    bounds[0] = float2(-INFINITY, INFINITY);
    segStart0[0] = 0;
    buildCount0[0] = p.rows;
    activeTotal0[0] = p.rows;
    numDerive[0] = 0;
    histArgs0[0] = (p.rows + p.samplesPerGroup - 1) / p.samplesPerGroup;
    histArgs0[1] = p.numTiles;
    histArgs0[2] = 1;
    routeArgs0[0] = (p.rows + TG_SIZE - 1) / TG_SIZE;
    routeArgs0[1] = 1;
    routeArgs0[2] = 1;
}

struct DecideParams {
    uint levelStart; uint numLevel; uint numFeatures; uint numBins;
    uint numTiles; uint samplesPerGroup; uint isLastLevel;
    float lambda; float learningRate; float minSplitGain; float catSmooth;
};

// GPU-side split decisions: one threadgroup per level. Phase 1 (parallel
// over nodes) reduces the per-feature split candidates, writes NodeSplit /
// leaf values / categorical masks, and derives child stats. Phase 2 (thread
// 0, tiny serial scan over <= 2^11 nodes) lays out the NEXT level's
// segments (every live child gets one; the smaller sibling additionally
// gets a histogram build), the subtraction map and both indirect dispatches.
// This removes every per-level CPU sync: a whole tree is one command buffer.
kernel void decide_splits(
    device const SplitResult *results       [[buffer(0)]],
    device const float       *hist          [[buffer(1)]],
    device NodeSplit         *splits        [[buffer(2)]],
    device float             *leafValues    [[buffer(3)]],
    device uint              *catMask       [[buffer(4)]],
    device float4            *stats         [[buffer(5)]],  // abs node: (G,H,count)
    device uint              *segStartNext  [[buffer(6)]],
    device uint              *cursorsNext   [[buffer(7)]],
    device uint              *buildCountNext [[buffer(8)]],
    device uint              *deriveMap     [[buffer(9)]],
    device uint              *numDerive     [[buffer(10)]],
    device uint              *activeTotalNext [[buffer(11)]],
    device uint              *histArgsNext  [[buffer(12)]],
    device uint              *routeArgsNext [[buffer(13)]],
    device float             *nodeGains     [[buffer(14)]],
    device float2            *bounds        [[buffer(15)]],
    device const char        *monotone      [[buffer(16)]],
    constant DecideParams    &p             [[buffer(17)]],
    uint tid [[thread_position_in_threadgroup]])
{
    // Root totals from feature 0's histogram (all bins, missing included).
    if (p.levelStart == 0 && tid == 0) {
        float G = 0.0f, H = 0.0f, C = 0.0f;
        for (uint b = 0; b < p.numBins; ++b) {
            G += hist[b*HIST_CH]; H += hist[b*HIST_CH+1]; C += hist[b*HIST_CH+2];
        }
        stats[0] = float4(G, H, C, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint local = tid; local < p.numLevel; local += TG_SIZE) {
        uint a = p.levelStart + local;
        float4 st = stats[a];
        NodeSplit leaf; leaf.feature = -1; leaf.bin = 0; leaf.flags = 0;
        if (st.z < 1.0f) {                       // dead node
            splits[a] = leaf; leafValues[a] = 0.0f; nodeGains[a] = 0.0f;
            continue;
        }
        int bestF = -1;
        SplitResult best;
        best.gain = -INFINITY; best.bin = 0; best.gl = 0; best.hl = 0; best.flags = 0;
        for (uint f = 0; f < p.numFeatures; ++f) {
            SplitResult r = results[local * p.numFeatures + f];
            if (r.gain > best.gain) { best = r; bestF = int(f); }
        }
        if (bestF >= 0 && isfinite(best.gain) && best.gain > p.minSplitGain) {
            NodeSplit ns; ns.feature = bestF; ns.bin = best.bin; ns.flags = best.flags;
            splits[a] = ns; leafValues[a] = 0.0f; nodeGains[a] = best.gain;
            device const float *h =
                hist + (((ulong)local * p.numFeatures + uint(bestF)) * p.numBins) * HIST_CH;
            float leftCount = 0.0f;
            if (best.flags & FLAG_CATEGORICAL) {
                ushort order_[256]; float key[256];
                cat_sort(h, p.numBins, p.catSmooth, order_, key);
                uint mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (uint i2 = 0; i2 < best.bin; ++i2) {
                    uint b = order_[i2];
                    mask[b >> 5] |= (1u << (b & 31u));
                    leftCount += h[b*HIST_CH+2];
                }
                for (uint w = 0; w < 8; ++w) catMask[a*8u + w] = mask[w];
            } else {
                for (uint b = 0; b <= best.bin; ++b) leftCount += h[b*HIST_CH+2];
                if (best.flags & FLAG_DEFAULT_LEFT)
                    leftCount += h[(p.numBins-1)*HIST_CH+2];
            }
            stats[2*a+1] = float4(best.gl, best.hl, leftCount, 0.0f);
            stats[2*a+2] = float4(st.x - best.gl, st.y - best.hl, st.z - leftCount, 0.0f);
            // Monotone bounds propagation (LightGBM basic mode): children
            // inherit the parent's bounds; a split on a constrained numeric
            // feature additionally caps both sides at the midpoint of the
            // two child values.
            float2 pb = bounds[a];
            float2 lb = pb, rb = pb;
            int cst = (best.flags & FLAG_CATEGORICAL) ? 0 : int(monotone[bestF]);
            if (cst != 0) {
                float vl = -best.gl / (best.hl + p.lambda);
                float vr = -(st.x - best.gl) / (st.y - best.hl + p.lambda);
                // Clamp the midpoint into the parent's bounds FIRST: nested
                // constrained splits can otherwise drift the midpoint past
                // an inherited bound, crossing the child interval (lo > hi)
                // and silently discarding the ancestor's constraint.
                float mid = clamp(0.5f * (vl + vr), pb.x, pb.y);
                if (cst > 0) { lb.y = min(lb.y, mid); rb.x = max(rb.x, mid); }
                else         { lb.x = max(lb.x, mid); rb.y = min(rb.y, mid); }
            }
            bounds[2*a+1] = lb;
            bounds[2*a+2] = rb;
        } else {
            splits[a] = leaf;
            float2 bb = bounds[a];
            nodeGains[a] = 0.0f;
            leafValues[a] = p.learningRate
                * clamp(-st.x / (st.y + p.lambda), bb.x, bb.y);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid != 0 || p.isLastLevel != 0) return;
    uint nextStart = 2 * p.levelStart + 1;
    uint numNext = p.numLevel * 2;
    uint start = 0, maxBuild = 0, nd = 0;
    for (uint cl = 0; cl < numNext; ++cl) {
        uint cAbs = nextStart + cl;
        uint pAbs = (cAbs - 1) >> 1;
        bool live = splits[pAbs].feature >= 0;
        uint cnt = live ? uint(stats[cAbs].z) : 0u;
        segStartNext[cl] = start;                 // dead: duplicated start
        cursorsNext[cl] = start;
        start += cnt;
        // The smaller sibling builds its histogram; ties go left. The other
        // is derived by subtraction.
        uint sibAbs = (cAbs & 1u) == 1u ? cAbs + 1 : cAbs - 1;
        uint sibCnt = live ? uint(stats[sibAbs].z) : 0u;
        bool builds = live &&
            ((cAbs & 1u) == 1u ? cnt <= sibCnt : cnt < sibCnt);
        buildCountNext[cl] = builds ? cnt : 0u;
        if (builds) maxBuild = max(maxBuild, cnt);
        if (live && !builds) {
            deriveMap[nd*3] = cl;
            deriveMap[nd*3+1] = pAbs - p.levelStart;
            deriveMap[nd*3+2] = sibAbs - nextStart;
            ++nd;
        }
    }
    numDerive[0] = nd;
    activeTotalNext[0] = start;
    histArgsNext[0] = (maxBuild + p.samplesPerGroup - 1) / p.samplesPerGroup;
    histArgsNext[1] = p.numTiles;
    histArgsNext[2] = numNext;
    routeArgsNext[0] = (start + TG_SIZE - 1) / TG_SIZE;
    routeArgsNext[1] = 1;
    routeArgsNext[2] = 1;
}

struct FinalParams { uint lastStart; uint numLast; float lambda; float learningRate; };

// Nodes at the deepest level are always leaves; their stats were written by
// the last decide_splits pass.
kernel void final_leaves(
    device NodeSplit    *splits     [[buffer(0)]],
    device float        *leafValues [[buffer(1)]],
    device const float4 *stats      [[buffer(2)]],
    device const float2 *bounds     [[buffer(3)]],
    constant FinalParams &p         [[buffer(4)]],
    uint local [[thread_position_in_grid]])
{
    if (local >= p.numLast) return;
    uint a = p.lastStart + local;
    NodeSplit ns; ns.feature = -1; ns.bin = 0; ns.flags = 0;
    splits[a] = ns;
    float4 st = stats[a];
    float2 bb = bounds[a];
    leafValues[a] = st.z >= 1.0f
        ? p.learningRate * clamp(-st.x / (st.y + p.lambda), bb.x, bb.y) : 0.0f;
}

// Copy per-node sample counts out of the stats buffer so the host can
// attach cover to each finished tree (feature importance, SHAP weights).
kernel void copy_cover(
    device const float4 *stats [[buffer(0)]],
    device float        *cover [[buffer(1)]],
    constant uint       &n     [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) cover[i] = stats[i].z;
}

kernel void debug_bounds(
    device const float2 *bounds [[buffer(0)]],
    device float        *outHi  [[buffer(1)]],
    constant uint       &n      [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) outHi[i] = bounds[i].y;
}

struct ForestParams {
    uint numSamples; uint numFeatures; uint numTrees; uint nodesPerTree;
    uint dataBins; uint numClasses; uint transform;
};

// Batch inference over RAW feature values (no binning needed): walk every
// tree of the flattened forest per sample. Matches the CPU predict path's
// float semantics exactly (leaf sum accumulated first, base added last).
kernel void predict_forest(
    device const float *X          [[buffer(0)]],   // feature-major raw
    device const int   *feats      [[buffer(1)]],   // numTrees * nodesPerTree
    device const float *thresholds [[buffer(2)]],
    device const uchar *tflags     [[buffer(3)]],
    device const float *leaves     [[buffer(4)]],
    device const uint  *masks      [[buffer(5)]],   // per node 8 words (or 8 zeros)
    device const float *baseScores [[buffer(6)]],   // numClasses entries
    device float       *out        [[buffer(7)]],   // row-major i*numClasses+c
    constant ForestParams &p       [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.numSamples) return;
    for (uint c = 0; c < p.numClasses; ++c)
        out[i * p.numClasses + c] = baseScores[c];
    for (uint t = 0; t < p.numTrees; ++t) {
        uint b = t * p.nodesPerTree;
        uint n = 0;
        while (feats[b + n] >= 0) {
            uint f = uint(feats[b + n]);
            float v = X[(ulong)f * p.numSamples + i];
            bool left;
            uchar fl = tflags[b + n];
            if (fl & FLAG_CATEGORICAL) {
                uint cat = p.dataBins;                       // unseen -> missing
                if (isfinite(v)) {
                    float r = rint(v);
                    if (r >= 0.0f && uint(r) < p.dataBins) cat = uint(r);
                }
                left = (masks[(ulong)(b + n) * 8u + cat / 32u] >> (cat % 32u)) & 1u;
            } else if (isnan(v)) {
                left = (fl & FLAG_DEFAULT_LEFT) != 0;
            } else {
                left = v <= thresholds[b + n];
            }
            n = 2 * n + 1 + (left ? 0 : 1);
        }
        out[i * p.numClasses + t % p.numClasses] += leaves[b + n];
    }
    if (p.transform != 0) {
        for (uint c = 0; c < p.numClasses; ++c)
            out[i * p.numClasses + c] = exp(out[i * p.numClasses + c]);
    }
}

kernel void zero_buffer(
    device uint   *buf [[buffer(0)]],
    constant uint &n   [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) buf[i] = 0;
}

struct ShadowParams { uint numSamples; uint numFeatures; uint halfBits; uint seed; };

// Bijective pseudo-random permutation of [0, n): a 4-round Feistel network
// over the enclosing power-of-two domain with cycle-walking. Lets every
// thread compute its own permuted index — no sort, no stored index array.
inline uint shadow_perm(uint i, uint n, uint hb, uint seed) {
    uint mask = (1u << hb) - 1u;
    do {
        uint L = i >> hb, R = i & mask;
        for (uint r = 0; r < 4; ++r) {
            uint t = R;
            R = L ^ (wang_hash(R ^ (seed + r * 0x9E3779B9u)) & mask);
            L = t;
        }
        i = (L << hb) | R;
    } while (i >= n);
    return i;
}

// Boruta shadow features without materialising any data: copy the binned
// matrix into the first half of a double-width tiled matrix and fill the
// second half with row-permuted copies of each column (fresh permutation
// per column per round via the seed). Bin edges are permutation-invariant,
// so shadows never exist as raw values anywhere — GPU bin bytes only.
kernel void shadow_bins(
    device const uchar   *src [[buffer(0)]],   // tiled, numFeatures wide
    device uchar         *dst [[buffer(1)]],   // tiled, 2*numFeatures wide
    constant ShadowParams &p  [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x, f = gid.y;
    if (i >= p.numSamples || f >= p.numFeatures) return;
    dst[bin_index(f, i, p.numSamples)] = src[bin_index(f, i, p.numSamples)];
    uint j = shadow_perm(i, p.numSamples, p.halfBits,
                         p.seed ^ wang_hash(f + 1u));
    dst[bin_index(p.numFeatures + f, i, p.numSamples)] =
        src[bin_index(f, j, p.numSamples)];
}

// One root-to-leaf path split, precomputed on the host (GPUTreeSHAP-style
// path decomposition: the recursive TreeSHAP sum distributes over leaves).
struct ShapElement {
    uint  feature;
    float zeroFraction;   // cover(child)/cover(parent) along the path
    float threshold;
    uint  flags;          // FLAG_* bits; bit 8 = path continues to LEFT child
    uint  nodeGlobal;     // tree*nodesPerTree + node, for catMask lookup
};
#define SHAP_CHILD_LEFT 0x100u

struct ShapParams {
    uint totalRows; uint rowCount; uint rowOffset;
    uint numPaths; uint numClasses; uint numFeatures; uint dataBins;
};

// Exact TreeSHAP, one thread per (row, leaf path). Duplicate features on a
// path are merged multiplicatively up front — EXTEND is commutative
// polynomial multiplication, so pre-merging equals the recursive
// algorithm's unwind/re-extend. Path length <= 12 (maxDepth cap), so the
// whole computation fits in registers; contributions land in the output
// via float atomics.
#define SHAP_MAX_PATH 13
kernel void gpu_treeshap(
    device const float       *X         [[buffer(0)]],   // feature-major raw
    device const ShapElement *elements  [[buffer(1)]],
    device const uint        *pathStart [[buffer(2)]],   // numPaths + 1
    device const float       *pathLeaf  [[buffer(3)]],
    device const uint        *pathClass [[buffer(4)]],
    device const uint        *masks     [[buffer(5)]],   // per node 8 words
    device atomic_float      *out       [[buffer(6)]],   // rows*K*(F+1)
    constant ShapParams      &p         [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.rowCount || gid.y >= p.numPaths) return;
    uint i = p.rowOffset + gid.x;

    uint  ufeat[SHAP_MAX_PATH];
    float uzero[SHAP_MAX_PATH];
    float uone[SHAP_MAX_PATH];
    uint U = 0;
    for (uint e = pathStart[gid.y]; e < pathStart[gid.y + 1]; ++e) {
        ShapElement el = elements[e];
        float v = X[(ulong)el.feature * p.totalRows + i];
        bool left;
        if (el.flags & FLAG_CATEGORICAL) {
            uint cat = p.dataBins;                       // unseen -> missing
            if (isfinite(v)) {
                float r = rint(v);
                if (r >= 0.0f && uint(r) < p.dataBins) cat = uint(r);
            }
            left = (masks[(ulong)el.nodeGlobal * 8u + cat / 32u] >> (cat % 32u)) & 1u;
        } else if (isnan(v)) {
            left = (el.flags & FLAG_DEFAULT_LEFT) != 0;
        } else {
            left = v <= el.threshold;
        }
        float one = (left == ((el.flags & SHAP_CHILD_LEFT) != 0)) ? 1.0f : 0.0f;
        int k = -1;
        for (uint j = 0; j < U; ++j)
            if (ufeat[j] == el.feature) { k = int(j); break; }
        if (k >= 0) { uzero[k] *= el.zeroFraction; uone[k] *= one; }
        else { ufeat[U] = el.feature; uzero[U] = el.zeroFraction; uone[U] = one; ++U; }
    }
    if (U == 0) return;

    // EXTEND: permutation-weight coefficients over the merged path, with
    // the standard dummy element (1, 1) at index 0.
    float pw[SHAP_MAX_PATH + 1];
    pw[0] = 1.0f;
    for (uint j = 1; j <= U; ++j) {
        float one = uone[j - 1], zero = uzero[j - 1];
        pw[j] = 0.0f;
        for (int q = int(j) - 1; q >= 0; --q) {
            pw[q + 1] += one * pw[q] * float(q + 1) / float(j + 1);
            pw[q]      = zero * pw[q] * float(int(j) - q) / float(j + 1);
        }
    }

    float leafV = pathLeaf[gid.y];
    ulong outBase = ((ulong)i * p.numClasses + pathClass[gid.y])
                    * (ulong)(p.numFeatures + 1);
    for (uint idx = 1; idx <= U; ++idx) {
        float one = uone[idx - 1], zero = uzero[idx - 1];
        float total = 0.0f;
        if (one != 0.0f) {
            float next = pw[U];
            for (int q = int(U) - 1; q >= 0; --q) {
                float tmp = next / (float(q + 1) * one);
                total += tmp;
                next = pw[q] - tmp * zero * float(int(U) - q);
            }
        } else if (zero != 0.0f) {
            for (int q = int(U) - 1; q >= 0; --q)
                total += pw[q] / (zero * float(int(U) - q));
        }
        float phi = total * float(U + 1) * (one - zero) * leafV;
        if (phi != 0.0f)
            atomic_fetch_add_explicit(&out[outBase + ufeat[idx - 1]], phi,
                                      memory_order_relaxed);
    }
}

struct PredictParams { uint numSamples; uint maxDepth; uint numBins; uint predOffset; };

// Assign each sample its leaf node id for the finished tree (leaf renewal
// for L1/quantile objectives reads these to compute per-leaf residual
// quantiles on the host).
kernel void assign_leaves(
    device const uchar     *bins       [[buffer(0)]],
    device const NodeSplit *splits     [[buffer(1)]],
    device const uint      *catMask    [[buffer(2)]],
    device int             *leafIdx    [[buffer(3)]],
    constant PredictParams &p          [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.numSamples) return;
    uint n = 0;
    for (uint d = 0; d < p.maxDepth; ++d) {
        NodeSplit s = splits[n];
        if (s.feature < 0) break;
        uchar b = bins[bin_index(uint(s.feature), i, p.numSamples)];
        n = 2 * n + 1 + (goes_left(b, s, p.numBins, catMask, n) ? 0 : 1);
    }
    leafIdx[i] = int(n);
}

// Walk one finished tree (in the nodeSplits/leafValues buffers) for every
// sample of a pre-binned matrix and add the leaf value to its prediction.
// Used for per-iteration validation-set evaluation.
kernel void predict_tree_binned(
    device const uchar     *bins       [[buffer(0)]],
    device float           *preds      [[buffer(1)]],
    device const NodeSplit *splits     [[buffer(2)]],
    device const float     *leafValues [[buffer(3)]],
    device const uint      *catMask    [[buffer(4)]],
    constant PredictParams &p          [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.numSamples) return;
    uint n = 0;
    for (uint d = 0; d < p.maxDepth; ++d) {
        NodeSplit s = splits[n];
        if (s.feature < 0) break;
        uchar b = bins[bin_index(uint(s.feature), i, p.numSamples)];
        n = 2 * n + 1 + (goes_left(b, s, p.numBins, catMask, n) ? 0 : 1);
    }
    preds[p.predOffset + i] += leafValues[n];
}
"""
