#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define MAGIC "LZTOK77"
#define MAGIC_SIZE 7
#define FORMAT_VERSION 1
#define MODE_BYTE 1
#define MODE_U32 2
#define MAX_WINDOW 1048576u
#define MAX_LOOKAHEAD 1024u
#define MAX_CANDIDATES 256u
#define HASH_BITS 20u
#define HASH_SIZE (1u << HASH_BITS)
#define HASH_MASK (HASH_SIZE - 1u)
#define NO_POS UINT32_MAX
#define LEN_CODES 32u
#define DIST_CODES 40u
#define END_SYMBOL_OFFSET 0u

typedef struct {
    uint32_t window_size;
    uint32_t lookahead_size;
    uint32_t min_match_length;
    uint32_t mode;
} Config;

typedef struct {
    uint32_t *symbols;
    uint8_t *bytes;
    uint64_t n;
    uint32_t alphabet_size;
    uint64_t source_bytes;
    uint32_t mode;
} Input;

typedef struct {
    uint64_t literals;
    uint64_t matches;
    uint64_t literal_bits;
    uint64_t length_bits;
    uint64_t distance_bits;
    uint64_t total_extra_bits;
} Stats;

typedef struct {
    uint8_t is_match;
    uint32_t literal;
    uint32_t length;
    uint32_t distance;
    uint64_t pos;
} Event;

typedef struct {
    uint64_t bits;
    uint8_t count;
    FILE *fp;
} BitWriter;

typedef struct {
    uint64_t bits;
    uint8_t count;
    FILE *fp;
} BitReader;

typedef struct {
    int32_t symbol;
    int32_t child[2];
} DecodeNode;

typedef struct {
    DecodeNode *nodes;
    uint32_t count;
    uint32_t cap;
} DecodeTree;

typedef struct {
    uint32_t symbol;
    uint8_t len;
} CanonItem;

typedef struct {
    uint64_t freq;
    int32_t left;
    int32_t right;
    uint32_t symbol;
    uint8_t is_leaf;
} HuffNode;

typedef struct {
    int32_t *items;
    uint32_t size;
    uint32_t cap;
    HuffNode *nodes;
} Heap;

typedef struct {
    FILE *fp;
    const uint64_t *ll_freq;
    const uint64_t *dist_freq;
    uint64_t *ll_bits;
    uint64_t *dist_bits;
    uint64_t extra_bits;
} EncodeCtx;

static const uint16_t len_base[LEN_CODES] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27,
    31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
    259, 515, 771
};
static const uint8_t len_extra[LEN_CODES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    8, 8, 8
};
static const uint32_t dist_base[DIST_CODES] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129,
    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
    6145, 8193, 12289, 16385, 24577, 32769, 49153, 65537, 98305,
    131073, 196609, 262145, 393217, 524289, 786433
};
static const uint8_t dist_extra[DIST_CODES] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
    6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
    14, 14, 15, 15, 16, 16, 17, 17, 18, 18
};

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "error: %s: %s\n", msg, strerror(errno));
    exit(1);
}

static void *xcalloc(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    if (!ptr) die_errno("calloc failed");
    return ptr;
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) die_errno("malloc failed");
    return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
    void *out = realloc(ptr, size);
    if (!out) die_errno("realloc failed");
    return out;
}

static uint32_t parse_u32(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(s, &end, 10);
    if (errno || !end || *end || value > UINT32_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint32_t)value;
}

static FILE *xfopen(const char *path, const char *mode) {
    FILE *fp = fopen(path, mode);
    if (!fp) {
        fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    return fp;
}

static uint64_t file_size(FILE *fp) {
    if (fseeko(fp, 0, SEEK_END) != 0) die_errno("fseeko failed");
    off_t size = ftello(fp);
    if (size < 0) die_errno("ftello failed");
    if (fseeko(fp, 0, SEEK_SET) != 0) die_errno("fseeko failed");
    return (uint64_t)size;
}

static uint32_t read_le32_buf(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void write_u8(FILE *fp, uint8_t v) {
    if (fputc(v, fp) == EOF) die_errno("write failed");
}

static void write_le32(FILE *fp, uint32_t v) {
    uint8_t b[4] = {v & 255u, (v >> 8) & 255u, (v >> 16) & 255u, (v >> 24) & 255u};
    if (fwrite(b, 1, 4, fp) != 4) die_errno("write failed");
}

static void write_le64(FILE *fp, uint64_t v) {
    for (int i = 0; i < 8; i++) write_u8(fp, (uint8_t)(v >> (i * 8)));
}

static uint8_t read_u8(FILE *fp) {
    int c = fgetc(fp);
    if (c == EOF) die("unexpected EOF");
    return (uint8_t)c;
}

static uint32_t read_le32(FILE *fp) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) die("unexpected EOF");
    return read_le32_buf(b);
}

static uint64_t read_le64(FILE *fp) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)read_u8(fp) << (i * 8);
    return v;
}

static Input read_input(const char *path, uint32_t mode) {
    FILE *fp = xfopen(path, "rb");
    uint64_t size = file_size(fp);
    Input in = {0};
    in.source_bytes = size;
    in.mode = mode;
    if (mode == MODE_BYTE) {
        if (size > UINT32_MAX) die("input has too many byte symbols for this build");
        uint8_t *buf = xmalloc((size_t)(size ? size : 1));
        if (size && fread(buf, 1, (size_t)size, fp) != size) die_errno("read failed");
        in.bytes = buf;
        in.n = size;
        in.alphabet_size = 256;
    } else {
        if (size % 4 != 0) die("u32 token input size must be a multiple of 4 bytes");
        in.n = size / 4;
        if (in.n > UINT32_MAX) die("input has too many token symbols for this build");
        uint8_t *buf = xmalloc((size_t)(size ? size : 1));
        if (size && fread(buf, 1, (size_t)size, fp) != size) die_errno("read failed");
        in.symbols = xmalloc((size_t)(in.n ? in.n : 1) * sizeof(uint32_t));
        uint32_t max_symbol = 0;
        for (uint64_t i = 0; i < in.n; i++) {
            uint32_t sym = read_le32_buf(buf + i * 4);
            in.symbols[i] = sym;
            if (sym > max_symbol) max_symbol = sym;
        }
        if (max_symbol == UINT32_MAX) die("token id UINT32_MAX is reserved");
        in.alphabet_size = max_symbol + 1;
        free(buf);
    }
    fclose(fp);
    return in;
}

static uint32_t symbol_at(const Input *in, uint64_t pos) {
    return in->mode == MODE_BYTE ? in->bytes[pos] : in->symbols[pos];
}

static uint32_t hash_at(const Input *in, uint64_t pos, uint32_t ngram) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < ngram; i++) {
        h ^= (uint64_t)symbol_at(in, pos + i) + 0x9e3779b97f4a7c15ull;
        h *= 1099511628211ull;
    }
    return (uint32_t)((h ^ (h >> 32)) & HASH_MASK);
}

static uint32_t length_code(uint32_t length, uint32_t *extra_value, uint8_t *extra_count) {
    for (uint32_t i = 0; i < LEN_CODES; i++) {
        uint32_t span = 1u << len_extra[i];
        if (length >= len_base[i] && length < len_base[i] + span) {
            *extra_value = length - len_base[i];
            *extra_count = len_extra[i];
            return i;
        }
    }
    die("length is outside DEFLATE range");
    return 0;
}

static uint32_t distance_code(uint32_t distance, uint32_t *extra_value, uint8_t *extra_count) {
    for (uint32_t i = 0; i < DIST_CODES; i++) {
        uint32_t span = 1u << dist_extra[i];
        if (distance >= dist_base[i] && distance < dist_base[i] + span) {
            *extra_value = distance - dist_base[i];
            *extra_count = dist_extra[i];
            return i;
        }
    }
    die("distance is outside DEFLATE range");
    return 0;
}

static void bw_init(BitWriter *bw, FILE *fp) {
    bw->bits = 0;
    bw->count = 0;
    bw->fp = fp;
}

static void bw_write(BitWriter *bw, uint64_t bits, uint8_t count) {
    for (int i = count - 1; i >= 0; i--) {
        bw->bits = (bw->bits << 1) | ((bits >> i) & 1u);
        bw->count++;
        if (bw->count == 8) {
            write_u8(bw->fp, (uint8_t)bw->bits);
            bw->bits = 0;
            bw->count = 0;
        }
    }
}

static void bw_flush(BitWriter *bw) {
    if (bw->count) {
        bw->bits <<= (8 - bw->count);
        write_u8(bw->fp, (uint8_t)bw->bits);
        bw->bits = 0;
        bw->count = 0;
    }
}

static void br_init(BitReader *br, FILE *fp) {
    br->bits = 0;
    br->count = 0;
    br->fp = fp;
}

static uint32_t br_read(BitReader *br, uint8_t count) {
    uint32_t out = 0;
    for (uint8_t i = 0; i < count; i++) {
        if (br->count == 0) {
            int c = fgetc(br->fp);
            if (c == EOF) die("unexpected EOF in bitstream");
            br->bits = (uint8_t)c;
            br->count = 8;
        }
        out = (out << 1) | ((br->bits >> 7) & 1u);
        br->bits <<= 1;
        br->count--;
    }
    return out;
}

static int heap_less(Heap *h, int32_t a, int32_t b) {
    HuffNode *na = &h->nodes[a], *nb = &h->nodes[b];
    if (na->freq != nb->freq) return na->freq < nb->freq;
    return na->symbol < nb->symbol;
}

static void heap_push(Heap *h, int32_t item) {
    if (h->size == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 256;
        h->items = xrealloc(h->items, h->cap * sizeof(int32_t));
    }
    uint32_t i = h->size++;
    h->items[i] = item;
    while (i) {
        uint32_t p = (i - 1) / 2;
        if (!heap_less(h, h->items[i], h->items[p])) break;
        int32_t tmp = h->items[i];
        h->items[i] = h->items[p];
        h->items[p] = tmp;
        i = p;
    }
}

static int32_t heap_pop(Heap *h) {
    int32_t out = h->items[0];
    h->items[0] = h->items[--h->size];
    uint32_t i = 0;
    for (;;) {
        uint32_t l = i * 2 + 1, r = l + 1, m = i;
        if (l < h->size && heap_less(h, h->items[l], h->items[m])) m = l;
        if (r < h->size && heap_less(h, h->items[r], h->items[m])) m = r;
        if (m == i) break;
        int32_t tmp = h->items[i];
        h->items[i] = h->items[m];
        h->items[m] = tmp;
        i = m;
    }
    return out;
}

static void assign_lengths(HuffNode *nodes, int32_t idx, uint8_t depth, uint8_t *lengths) {
    if (nodes[idx].is_leaf) {
        lengths[nodes[idx].symbol] = depth ? depth : 1;
        return;
    }
    if (depth >= 56) die("Huffman tree too deep for this implementation");
    assign_lengths(nodes, nodes[idx].left, depth + 1, lengths);
    assign_lengths(nodes, nodes[idx].right, depth + 1, lengths);
}

static void build_huffman_lengths(const uint64_t *freq, uint32_t n, uint8_t *lengths) {
    memset(lengths, 0, n);
    uint32_t nonzero = 0;
    for (uint32_t i = 0; i < n; i++) if (freq[i]) nonzero++;
    if (nonzero == 0) return;

    HuffNode *nodes = xcalloc(nonzero * 2 + 1, sizeof(HuffNode));
    Heap heap = {0};
    heap.nodes = nodes;
    int32_t node_count = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!freq[i]) continue;
        nodes[node_count].freq = freq[i];
        nodes[node_count].symbol = i;
        nodes[node_count].is_leaf = 1;
        heap_push(&heap, node_count++);
    }
    if (heap.size == 1) {
        lengths[nodes[heap.items[0]].symbol] = 1;
    } else {
        while (heap.size > 1) {
            int32_t a = heap_pop(&heap);
            int32_t b = heap_pop(&heap);
            nodes[node_count].freq = nodes[a].freq + nodes[b].freq;
            nodes[node_count].symbol = nodes[a].symbol < nodes[b].symbol ? nodes[a].symbol : nodes[b].symbol;
            nodes[node_count].left = a;
            nodes[node_count].right = b;
            nodes[node_count].is_leaf = 0;
            heap_push(&heap, node_count++);
        }
        assign_lengths(nodes, heap.items[0], 0, lengths);
    }
    free(heap.items);
    free(nodes);
}

static int canon_cmp(const void *a, const void *b) {
    const CanonItem *ia = a, *ib = b;
    if (ia->len != ib->len) return (int)ia->len - (int)ib->len;
    return ia->symbol < ib->symbol ? -1 : ia->symbol > ib->symbol;
}

static void build_canonical_codes(const uint8_t *lengths, uint32_t n, uint64_t *codes) {
    CanonItem *items = xmalloc((size_t)n * sizeof(CanonItem));
    uint32_t count = 0;
    for (uint32_t i = 0; i < n; i++) {
        codes[i] = 0;
        if (lengths[i]) {
            items[count].symbol = i;
            items[count].len = lengths[i];
            count++;
        }
    }
    qsort(items, count, sizeof(CanonItem), canon_cmp);
    uint64_t code = 0;
    uint8_t prev_len = 0;
    for (uint32_t i = 0; i < count; i++) {
        code <<= (items[i].len - prev_len);
        codes[items[i].symbol] = code;
        code++;
        prev_len = items[i].len;
    }
    free(items);
}

static void dt_add_node(DecodeTree *tree) {
    if (tree->count == tree->cap) {
        tree->cap = tree->cap ? tree->cap * 2 : 1024;
        tree->nodes = xrealloc(tree->nodes, tree->cap * sizeof(DecodeNode));
    }
    tree->nodes[tree->count].symbol = -1;
    tree->nodes[tree->count].child[0] = -1;
    tree->nodes[tree->count].child[1] = -1;
    tree->count++;
}

static DecodeTree build_decode_tree(const uint8_t *lengths, uint32_t n) {
    uint64_t *codes = xcalloc(n, sizeof(uint64_t));
    build_canonical_codes(lengths, n, codes);
    DecodeTree tree = {0};
    dt_add_node(&tree);
    for (uint32_t sym = 0; sym < n; sym++) {
        uint8_t len = lengths[sym];
        if (!len) continue;
        int32_t node = 0;
        for (int i = len - 1; i >= 0; i--) {
            uint32_t bit = (uint32_t)((codes[sym] >> i) & 1u);
            if (tree.nodes[node].child[bit] < 0) {
                tree.nodes[node].child[bit] = (int32_t)tree.count;
                dt_add_node(&tree);
            }
            node = tree.nodes[node].child[bit];
        }
        tree.nodes[node].symbol = (int32_t)sym;
    }
    free(codes);
    return tree;
}

static uint32_t decode_symbol(BitReader *br, DecodeTree *tree) {
    int32_t node = 0;
    while (tree->nodes[node].symbol < 0) {
        uint32_t bit = br_read(br, 1);
        node = tree->nodes[node].child[bit];
        if (node < 0) die("invalid Huffman code");
    }
    return (uint32_t)tree->nodes[node].symbol;
}

typedef void (*EventFn)(const Event *ev, void *ctx);

static void free_input(Input *in) {
    free(in->symbols);
    free(in->bytes);
    in->symbols = NULL;
    in->bytes = NULL;
}

static void insert_pos(const Input *in, uint32_t *head, uint32_t *prev, uint64_t pos, uint32_t ngram) {
    uint64_t n = in->n;
    if (pos + ngram > n) return;
    uint32_t h = hash_at(in, pos, ngram);
    prev[pos] = head[h];
    head[h] = (uint32_t)pos;
}

static void parse_lz77(const Input *in, const Config *cfg, EventFn emit, void *ctx) {
    uint32_t ngram = cfg->min_match_length;
    uint32_t *head = xmalloc(HASH_SIZE * sizeof(uint32_t));
    uint32_t *prev = xmalloc((size_t)(in->n ? in->n : 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i < HASH_SIZE; i++) head[i] = NO_POS;
    for (uint64_t i = 0; i < in->n; i++) prev[i] = NO_POS;

    uint64_t pos = 0;
    while (pos < in->n) {
        uint32_t best_len = 0;
        uint32_t best_dist = 0;
        uint64_t remaining = in->n - pos;
        if (remaining >= cfg->min_match_length) {
            uint32_t h = hash_at(in, pos, ngram);
            uint32_t cand = head[h];
            uint32_t checked = 0;
            while (cand != NO_POS && checked < MAX_CANDIDATES) {
                uint32_t dist = (uint32_t)(pos - cand);
                if (dist > cfg->window_size) break;
                uint32_t max_len = cfg->lookahead_size;
                if (max_len > remaining) max_len = (uint32_t)remaining;
                uint32_t len = 0;
                while (len < max_len && symbol_at(in, cand + len) == symbol_at(in, pos + len)) len++;
                if (len > best_len) {
                    best_len = len;
                    best_dist = dist;
                    if (best_len == max_len) break;
                }
                cand = prev[cand];
                checked++;
            }
        }

        if (best_len >= cfg->min_match_length) {
            Event ev = {.is_match = 1, .length = best_len, .distance = best_dist, .pos = pos};
            emit(&ev, ctx);
            for (uint32_t k = 0; k < best_len; k++) insert_pos(in, head, prev, pos + k, ngram);
            pos += best_len;
        } else {
            Event ev = {.is_match = 0, .literal = symbol_at(in, pos), .pos = pos};
            emit(&ev, ctx);
            insert_pos(in, head, prev, pos, ngram);
            pos++;
        }
    }
    free(head);
    free(prev);
}

typedef struct {
    uint64_t *ll_freq;
    uint64_t *dist_freq;
    uint32_t alphabet_size;
    uint64_t extra_bits;
    Stats stats;
} CountCtx;

static void freq_event(const Event *ev, void *ctx) {
    CountCtx *cc = ctx;
    if (ev->is_match) {
        uint32_t extra_value;
        uint8_t extra_count;
        uint32_t lc = length_code(ev->length, &extra_value, &extra_count);
        uint32_t dc = distance_code(ev->distance, &extra_value, &extra_count);
        cc->ll_freq[cc->alphabet_size + lc]++;
        cc->dist_freq[dc]++;
        cc->extra_bits += len_extra[lc] + dist_extra[dc];
        cc->stats.matches++;
        cc->stats.total_extra_bits += len_extra[lc] + dist_extra[dc];
    } else {
        cc->ll_freq[ev->literal]++;
        cc->stats.literals++;
    }
}

typedef struct {
    BitWriter *bw;
    uint64_t *ll_codes;
    uint8_t *ll_lens;
    uint64_t *dist_codes;
    uint8_t *dist_lens;
    uint32_t alphabet_size;
} WriteCtx;

static void write_event(const Event *ev, void *ctx) {
    WriteCtx *wc = ctx;
    if (ev->is_match) {
        uint32_t extra_value;
        uint8_t extra_count;
        uint32_t lc = length_code(ev->length, &extra_value, &extra_count);
        uint32_t sym = wc->alphabet_size + lc;
        bw_write(wc->bw, wc->ll_codes[sym], wc->ll_lens[sym]);
        if (extra_count) bw_write(wc->bw, extra_value, extra_count);
        uint32_t dc = distance_code(ev->distance, &extra_value, &extra_count);
        bw_write(wc->bw, wc->dist_codes[dc], wc->dist_lens[dc]);
        if (extra_count) bw_write(wc->bw, extra_value, extra_count);
    } else {
        bw_write(wc->bw, wc->ll_codes[ev->literal], wc->ll_lens[ev->literal]);
    }
}

static uint64_t huffman_cost(const uint64_t *freq, const uint8_t *lens, uint32_t n) {
    uint64_t bits = 0;
    for (uint32_t i = 0; i < n; i++) bits += freq[i] * lens[i];
    return bits;
}

static void validate_config(Config *cfg) {
    if (cfg->window_size < 1 || cfg->window_size > MAX_WINDOW) die("window-size must be in 1..32768");
    if (cfg->lookahead_size < 3 || cfg->lookahead_size > MAX_LOOKAHEAD) die("lookahead-size must be in 3..258");
    if (cfg->min_match_length < 3 || cfg->min_match_length > cfg->lookahead_size) {
        die("min-match-length must be in 3..lookahead-size");
    }
}

static void write_header(FILE *fp, const Config *cfg, const Input *in, uint32_t ll_count,
                         const uint8_t *ll_lens, const uint8_t *dist_lens) {
    if (fwrite(MAGIC, 1, MAGIC_SIZE, fp) != MAGIC_SIZE) die_errno("write failed");
    write_u8(fp, FORMAT_VERSION);
    write_u8(fp, (uint8_t)cfg->mode);
    write_le32(fp, cfg->window_size);
    write_le32(fp, cfg->lookahead_size);
    write_le32(fp, cfg->min_match_length);
    write_le64(fp, in->n);
    write_le64(fp, in->source_bytes);
    write_le32(fp, in->alphabet_size);
    write_le32(fp, ll_count);
    if (fwrite(ll_lens, 1, ll_count, fp) != ll_count) die_errno("write failed");
    if (fwrite(dist_lens, 1, DIST_CODES, fp) != DIST_CODES) die_errno("write failed");
}

static void read_header(FILE *fp, Config *cfg, uint64_t *n_symbols, uint64_t *source_bytes,
                        uint32_t *alphabet_size, uint32_t *ll_count,
                        uint8_t **ll_lens, uint8_t **dist_lens) {
    char magic[MAGIC_SIZE];
    if (fread(magic, 1, MAGIC_SIZE, fp) != MAGIC_SIZE || memcmp(magic, MAGIC, MAGIC_SIZE) != 0) {
        die("not an lztok77 file");
    }
    uint8_t version = read_u8(fp);
    if (version != FORMAT_VERSION) die("unsupported format version");
    cfg->mode = read_u8(fp);
    cfg->window_size = read_le32(fp);
    cfg->lookahead_size = read_le32(fp);
    cfg->min_match_length = read_le32(fp);
    validate_config(cfg);
    *n_symbols = read_le64(fp);
    *source_bytes = read_le64(fp);
    *alphabet_size = read_le32(fp);
    *ll_count = read_le32(fp);
    if (*ll_count != *alphabet_size + LEN_CODES) die("invalid literal/length table size");
    *ll_lens = xmalloc(*ll_count);
    *dist_lens = xmalloc(DIST_CODES);
    if (fread(*ll_lens, 1, *ll_count, fp) != *ll_count) die("unexpected EOF");
    if (fread(*dist_lens, 1, DIST_CODES, fp) != DIST_CODES) die("unexpected EOF");
}

static void command_compress(const char *input_path, const char *output_path, Config cfg) {
    validate_config(&cfg);
    Input in = read_input(input_path, cfg.mode);
    uint32_t ll_count = in.alphabet_size + LEN_CODES;
    uint64_t *ll_freq = xcalloc(ll_count, sizeof(uint64_t));
    uint64_t *dist_freq = xcalloc(DIST_CODES, sizeof(uint64_t));
    CountCtx cc = {.ll_freq = ll_freq, .dist_freq = dist_freq, .alphabet_size = in.alphabet_size};
    parse_lz77(&in, &cfg, freq_event, &cc);

    uint8_t *ll_lens = xcalloc(ll_count, 1);
    uint8_t *dist_lens = xcalloc(DIST_CODES, 1);
    build_huffman_lengths(ll_freq, ll_count, ll_lens);
    build_huffman_lengths(dist_freq, DIST_CODES, dist_lens);
    uint64_t *ll_codes = xcalloc(ll_count, sizeof(uint64_t));
    uint64_t *dist_codes = xcalloc(DIST_CODES, sizeof(uint64_t));
    build_canonical_codes(ll_lens, ll_count, ll_codes);
    build_canonical_codes(dist_lens, DIST_CODES, dist_codes);

    FILE *out = xfopen(output_path, "wb");
    write_header(out, &cfg, &in, ll_count, ll_lens, dist_lens);
    BitWriter bw;
    bw_init(&bw, out);
    WriteCtx wc = {
        .bw = &bw,
        .ll_codes = ll_codes,
        .ll_lens = ll_lens,
        .dist_codes = dist_codes,
        .dist_lens = dist_lens,
        .alphabet_size = in.alphabet_size,
    };
    parse_lz77(&in, &cfg, write_event, &wc);
    bw_flush(&bw);
    fclose(out);

    uint64_t ll_bits = huffman_cost(ll_freq, ll_lens, ll_count);
    uint64_t dist_bits = huffman_cost(dist_freq, dist_lens, DIST_CODES);
    uint64_t total_bits = ll_bits + dist_bits + cc.extra_bits;
    printf("{\"mode\":\"%s\",\"input_symbols\":%" PRIu64 ",\"source_bytes\":%" PRIu64
           ",\"alphabet_size\":%" PRIu32 ",\"literals\":%" PRIu64 ",\"matches\":%" PRIu64
           ",\"huffman_bits\":%" PRIu64 ",\"extra_bits\":%" PRIu64
           ",\"estimated_payload_bits\":%" PRIu64 ",\"bits_per_source_byte\":%.6f}\n",
           cfg.mode == MODE_BYTE ? "byte" : "u32",
           in.n, in.source_bytes, in.alphabet_size, cc.stats.literals, cc.stats.matches,
           ll_bits + dist_bits, cc.extra_bits, total_bits,
           in.source_bytes ? (double)total_bits / (double)in.source_bytes : 0.0);

    free_input(&in);
    free(ll_freq); free(dist_freq); free(ll_lens); free(dist_lens); free(ll_codes); free(dist_codes);
}

static void write_output_symbol(FILE *out, uint32_t mode, uint32_t sym) {
    if (mode == MODE_BYTE) {
        if (sym > 255) die("decoded byte symbol outside byte range");
        write_u8(out, (uint8_t)sym);
    } else {
        write_le32(out, sym);
    }
}

static void command_decompress(const char *input_path, const char *output_path) {
    FILE *infile = xfopen(input_path, "rb");
    Config cfg = {0};
    uint64_t n_symbols, source_bytes;
    uint32_t alphabet_size, ll_count;
    uint8_t *ll_lens, *dist_lens;
    read_header(infile, &cfg, &n_symbols, &source_bytes, &alphabet_size, &ll_count, &ll_lens, &dist_lens);
    DecodeTree ll_tree = build_decode_tree(ll_lens, ll_count);
    DecodeTree dist_tree = build_decode_tree(dist_lens, DIST_CODES);
    FILE *out = xfopen(output_path, "wb");
    uint32_t *history = xmalloc((size_t)cfg.window_size * sizeof(uint32_t));
    uint64_t produced = 0;
    BitReader br;
    br_init(&br, infile);
    while (produced < n_symbols) {
        uint32_t sym = decode_symbol(&br, &ll_tree);
        if (sym < alphabet_size) {
            history[produced % cfg.window_size] = sym;
            produced++;
            write_output_symbol(out, cfg.mode, sym);
        } else {
            uint32_t lc = sym - alphabet_size;
            if (lc >= LEN_CODES) die("invalid length code");
            uint32_t length = len_base[lc] + (len_extra[lc] ? br_read(&br, len_extra[lc]) : 0);
            uint32_t dc = decode_symbol(&br, &dist_tree);
            if (dc >= DIST_CODES) die("invalid distance code");
            uint32_t distance = dist_base[dc] + (dist_extra[dc] ? br_read(&br, dist_extra[dc]) : 0);
            if (distance == 0 || distance > produced) die("invalid match distance");
            if (produced + length > n_symbols) die("match overruns output");
            for (uint32_t i = 0; i < length; i++) {
                uint32_t value = history[(produced - distance) % cfg.window_size];
                history[produced % cfg.window_size] = value;
                produced++;
                write_output_symbol(out, cfg.mode, value);
            }
        }
    }
    fclose(out);
    fclose(infile);
    free(history); free(ll_lens); free(dist_lens); free(ll_tree.nodes); free(dist_tree.nodes);
    (void)source_bytes;
}

static void command_analyze(const char *input_path, Config cfg) {
    validate_config(&cfg);
    Input in = read_input(input_path, cfg.mode);
    uint32_t ll_count = in.alphabet_size + LEN_CODES;
    uint64_t *ll_freq = xcalloc(ll_count, sizeof(uint64_t));
    uint64_t *dist_freq = xcalloc(DIST_CODES, sizeof(uint64_t));
    CountCtx cc = {.ll_freq = ll_freq, .dist_freq = dist_freq, .alphabet_size = in.alphabet_size};
    parse_lz77(&in, &cfg, freq_event, &cc);
    uint8_t *ll_lens = xcalloc(ll_count, 1);
    uint8_t *dist_lens = xcalloc(DIST_CODES, 1);
    build_huffman_lengths(ll_freq, ll_count, ll_lens);
    build_huffman_lengths(dist_freq, DIST_CODES, dist_lens);
    uint64_t ll_bits = huffman_cost(ll_freq, ll_lens, ll_count);
    uint64_t dist_bits = huffman_cost(dist_freq, dist_lens, DIST_CODES);
    uint64_t total_bits = ll_bits + dist_bits + cc.extra_bits;
    printf("{\"mode\":\"%s\",\"input_symbols\":%" PRIu64 ",\"source_bytes\":%" PRIu64
           ",\"alphabet_size\":%" PRIu32 ",\"window_size\":%" PRIu32
           ",\"lookahead_size\":%" PRIu32 ",\"min_match_length\":%" PRIu32
           ",\"literals\":%" PRIu64 ",\"matches\":%" PRIu64
           ",\"huffman_bits\":%" PRIu64 ",\"extra_bits\":%" PRIu64
           ",\"estimated_payload_bits\":%" PRIu64 ",\"bits_per_symbol\":%.6f"
           ",\"bits_per_source_byte\":%.6f}\n",
           cfg.mode == MODE_BYTE ? "byte" : "u32", in.n, in.source_bytes, in.alphabet_size,
           cfg.window_size, cfg.lookahead_size, cfg.min_match_length,
           cc.stats.literals, cc.stats.matches, ll_bits + dist_bits, cc.extra_bits, total_bits,
           in.n ? (double)total_bits / (double)in.n : 0.0,
           in.source_bytes ? (double)total_bits / (double)in.source_bytes : 0.0);
    free_input(&in); free(ll_freq); free(dist_freq); free(ll_lens); free(dist_lens);
}

typedef struct {
    uint64_t *length_counts;
    uint64_t *effective_counts;
    const uint32_t *token_byte_lengths;
    const uint32_t *token_byte_starts;
    const uint32_t *token_byte_ends;
    uint64_t token_count;
    uint32_t max_length;
    uint32_t max_effective;
    uint64_t matches;
    uint64_t literals;
} LengthHistCtx;

static void length_hist_event(const Event *ev, void *ctx) {
    LengthHistCtx *hc = ctx;
    if (!ev->is_match) {
        hc->literals++;
        return;
    }
    hc->matches++;
    if (ev->length <= hc->max_length) hc->length_counts[ev->length]++;
    if (hc->token_byte_starts && hc->token_byte_ends) {
        if (ev->pos + ev->length > hc->token_count) die("token byte offset sidecars are shorter than token input");
        uint64_t effective = (uint64_t)hc->token_byte_ends[ev->pos + ev->length - 1] -
                             (uint64_t)hc->token_byte_starts[ev->pos];
        if (effective > hc->max_effective) die("effective match length exceeds histogram limit");
        hc->effective_counts[effective]++;
    } else if (hc->token_byte_lengths) {
        uint64_t effective = 0;
        if (ev->pos + ev->length > hc->token_count) die("token byte-length sidecar is shorter than token input");
        for (uint32_t i = 0; i < ev->length; i++) effective += hc->token_byte_lengths[ev->pos + i];
        if (effective > hc->max_effective) die("effective match length exceeds histogram limit");
        hc->effective_counts[effective]++;
    }
}

static uint32_t *read_u32_sidecar(const char *path, uint64_t expected_count) {
    FILE *fp = xfopen(path, "rb");
    uint64_t size = file_size(fp);
    if (size % 4 != 0) die("u32 sidecar size must be a multiple of 4 bytes");
    uint64_t count = size / 4;
    if (count != expected_count) die("u32 sidecar length does not match input symbol count");
    uint8_t *buf = xmalloc((size_t)(size ? size : 1));
    if (size && fread(buf, 1, (size_t)size, fp) != size) die_errno("read failed");
    fclose(fp);
    uint32_t *values = xmalloc((size_t)(count ? count : 1) * sizeof(uint32_t));
    for (uint64_t i = 0; i < count; i++) values[i] = read_le32_buf(buf + i * 4);
    free(buf);
    return values;
}

static void write_hist_csv(const char *output_path, const LengthHistCtx *hc) {
    FILE *out = xfopen(output_path, "wb");
    fprintf(out, "kind,length,count,probability\n");
    for (uint32_t i = 0; i <= hc->max_length; i++) {
        if (!hc->length_counts[i]) continue;
        fprintf(out, "symbols,%" PRIu32 ",%" PRIu64 ",%.17g\n",
                i, hc->length_counts[i], hc->matches ? (double)hc->length_counts[i] / (double)hc->matches : 0.0);
    }
    if (hc->effective_counts) {
        for (uint32_t i = 0; i <= hc->max_effective; i++) {
            if (!hc->effective_counts[i]) continue;
            fprintf(out, "effective,%" PRIu32 ",%" PRIu64 ",%.17g\n",
                    i, hc->effective_counts[i], hc->matches ? (double)hc->effective_counts[i] / (double)hc->matches : 0.0);
        }
    }
    fclose(out);
}

static void command_length_hist(const char *input_path, const char *output_path,
                                const char *token_byte_lengths_path,
                                const char *token_byte_starts_path,
                                const char *token_byte_ends_path,
                                Config cfg) {
    validate_config(&cfg);
    Input in = read_input(input_path, cfg.mode);
    uint32_t *token_byte_lengths = NULL;
    uint32_t *token_byte_starts = NULL;
    uint32_t *token_byte_ends = NULL;
    uint32_t max_effective = cfg.lookahead_size;
    if (token_byte_starts_path || token_byte_ends_path) {
        if (!token_byte_starts_path || !token_byte_ends_path) die("token byte starts and ends must be provided together");
        token_byte_starts = read_u32_sidecar(token_byte_starts_path, in.n);
        token_byte_ends = read_u32_sidecar(token_byte_ends_path, in.n);
        max_effective = 0;
        for (uint64_t i = 0; i < in.n; i++) {
            if (token_byte_ends[i] < token_byte_starts[i]) die("token byte end precedes start");
            uint32_t length = token_byte_ends[i] - token_byte_starts[i];
            if (length * cfg.lookahead_size > max_effective) max_effective = length * cfg.lookahead_size;
        }
    } else if (token_byte_lengths_path) {
        token_byte_lengths = read_u32_sidecar(token_byte_lengths_path, in.n);
        max_effective = cfg.lookahead_size * 16;
        for (uint64_t i = 0; i < in.n; i++) {
            if (token_byte_lengths[i] > 65535u) die("token byte length is unexpectedly large");
            if (token_byte_lengths[i] * cfg.lookahead_size > max_effective) {
                max_effective = token_byte_lengths[i] * cfg.lookahead_size;
            }
        }
    }
    LengthHistCtx hc = {
        .length_counts = xcalloc((size_t)cfg.lookahead_size + 1, sizeof(uint64_t)),
        .effective_counts = (token_byte_lengths || token_byte_starts) ? xcalloc((size_t)max_effective + 1, sizeof(uint64_t)) : NULL,
        .token_byte_lengths = token_byte_lengths,
        .token_byte_starts = token_byte_starts,
        .token_byte_ends = token_byte_ends,
        .token_count = in.n,
        .max_length = cfg.lookahead_size,
        .max_effective = max_effective,
    };
    parse_lz77(&in, &cfg, length_hist_event, &hc);
    write_hist_csv(output_path, &hc);
    printf("{\"input_symbols\":%" PRIu64 ",\"matches\":%" PRIu64
           ",\"literals\":%" PRIu64 ",\"output\":\"%s\"}\n",
           in.n, hc.matches, hc.literals, output_path);
    free_input(&in);
    free(token_byte_lengths);
    free(token_byte_starts);
    free(token_byte_ends);
    free(hc.length_counts);
    free(hc.effective_counts);
}

static uint32_t parse_mode(const char *s) {
    if (strcmp(s, "byte") == 0) return MODE_BYTE;
    if (strcmp(s, "u32") == 0 || strcmp(s, "token") == 0) return MODE_U32;
    die("mode must be byte or u32");
    return 0;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "usage:\n"
        "  lztok compress --input PATH --output PATH [--mode byte|u32] [--window-size N] [--lookahead-size N] [--min-match-length N]\n"
        "  lztok decompress --input PATH --output PATH\n"
        "  lztok analyze --input PATH [--mode byte|u32] [--window-size N] [--lookahead-size N] [--min-match-length N]\n"
        "  lztok length-hist --input PATH --output PATH [--mode byte|u32] [--token-byte-lengths PATH | --token-byte-starts PATH --token-byte-ends PATH] [--window-size N] [--lookahead-size N] [--min-match-length N]\n"
        "\n"
        "defaults: --mode byte --window-size 32768 --lookahead-size 258 --min-match-length 3\n"
        "u32/token mode reads and writes raw little-endian uint32 token IDs.\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    const char *cmd = argv[1];
    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        usage(stdout);
        return 0;
    }
    const char *input = NULL;
    const char *output = NULL;
    const char *token_byte_lengths = NULL;
    const char *token_byte_starts = NULL;
    const char *token_byte_ends = NULL;
    Config cfg = {.window_size = 32768, .lookahead_size = 258, .min_match_length = 3, .mode = MODE_BYTE};
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output = argv[++i];
        else if (strcmp(argv[i], "--token-byte-lengths") == 0 && i + 1 < argc) token_byte_lengths = argv[++i];
        else if (strcmp(argv[i], "--token-byte-starts") == 0 && i + 1 < argc) token_byte_starts = argv[++i];
        else if (strcmp(argv[i], "--token-byte-ends") == 0 && i + 1 < argc) token_byte_ends = argv[++i];
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) cfg.mode = parse_mode(argv[++i]);
        else if (strcmp(argv[i], "--window-size") == 0 && i + 1 < argc) cfg.window_size = parse_u32(argv[++i], "window-size");
        else if (strcmp(argv[i], "--lookahead-size") == 0 && i + 1 < argc) cfg.lookahead_size = parse_u32(argv[++i], "lookahead-size");
        else if (strcmp(argv[i], "--min-match-length") == 0 && i + 1 < argc) cfg.min_match_length = parse_u32(argv[++i], "min-match-length");
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }
    if (strcmp(cmd, "compress") == 0) {
        if (!input || !output) die("compress requires --input and --output");
        command_compress(input, output, cfg);
    } else if (strcmp(cmd, "decompress") == 0) {
        if (!input || !output) die("decompress requires --input and --output");
        command_decompress(input, output);
    } else if (strcmp(cmd, "analyze") == 0) {
        if (!input) die("analyze requires --input");
        command_analyze(input, cfg);
    } else if (strcmp(cmd, "length-hist") == 0) {
        if (!input || !output) die("length-hist requires --input and --output");
        command_length_hist(input, output, token_byte_lengths, token_byte_starts, token_byte_ends, cfg);
    } else {
        fprintf(stderr, "unknown command: %s\n", cmd);
        usage(stderr);
        return 2;
    }
    return 0;
}
