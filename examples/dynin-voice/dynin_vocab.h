// dynin_vocab.h -- reads the Dynin-Omni token-id layout off a loaded GGUF
// rather than trusting the checkpoint's config.json.
//
// The Dynin-Omni vocabulary is three ranges stacked on the text tokenizer:
//
//   text    [0, text_len)                       text_len = len(text_tokenizer) = 126372
//   image   [text_len, text_len + 8192)         MAGVITv2 codes 0..8191  -> ids 126372..134563
//   speech  [text_len + 8192, + 4096)           EMOVA units 0..4095     -> ids 134564..138659
//
// The checkpoint's config.json says llm_vocab_size=126464 and the GGUF has 138752 rows; both
// figures include PADDING. The reference implementation offsets image codes and speech units
// by len(text_tokenizer), which is 126372: in the HF tokenizer ids 126372 and up do not exist,
// and in the GGUF they are the pieces "[PAD126372]", "[PAD126373]", ... . Hardcoding the
// config.json figures (126464 / 134656) instead produces a systematic +92 shift on every
// image code and every speech unit, silently ungrounding any tool built on top of it.
//
// Nothing here is taken on trust: text_len is READ from the loaded GGUF vocabulary as the
// first id whose piece is exactly "[PAD<id>]", then cross-checked against the value the
// checkpoint's tokenizer reports. A vocabulary that disagrees is refused by name.
#pragma once

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dynin_vocab {

struct Map {
    int32_t text_len    = 0;     // len(text_tokenizer): first padded id
    int32_t image_base  = 0;     // == text_len
    int32_t image_n     = 8192;  // MAGVITv2 codebook
    int32_t speech_base = 0;     // == text_len + image_n
    int32_t speech_n    = 4096;  // EMOVA FSQ codebook
    int32_t n_vocab     = 0;     // rows in the loaded vocabulary
    bool    ok          = false;
    char    why[192]    = {0};
};

// True when the piece is exactly "[PAD<id>]" for this id (the converter's padding name).
inline bool is_pad_piece(const char * piece, int32_t id) {
    if (!piece || std::strncmp(piece, "[PAD", 4) != 0) return false;
    char * end = nullptr;
    const long v = std::strtol(piece + 4, &end, 10);
    return end && *end == ']' && end[1] == '\0' && v == (long) id;
}

// Reads the map off the loaded model's vocabulary. Refuses (ok=false, why filled) when the
// vocabulary has no padding boundary, when the boundary is not where the checkpoint's own
// tokenizer puts it, or when the speech range would run past the last row.
inline Map read(const llama_vocab * vocab, int32_t expected_text_len = 126372) {
    Map m;
    m.n_vocab = llama_vocab_n_tokens(vocab);
    int32_t first_pad = -1;
    for (int32_t id = 0; id < m.n_vocab; id++) {
        if (is_pad_piece(llama_vocab_get_text(vocab, id), id)) { first_pad = id; break; }
    }
    if (first_pad < 0) {
        std::snprintf(m.why, sizeof(m.why), "no [PAD<id>] boundary in a vocabulary of %d rows", m.n_vocab);
        return m;
    }
    if (first_pad != expected_text_len) {
        std::snprintf(m.why, sizeof(m.why), "first [PAD<id>] is id %d, the checkpoint's tokenizer says len(text_tokenizer)=%d",
                      first_pad, expected_text_len);
        return m;
    }
    m.text_len    = first_pad;
    m.image_base  = m.text_len;
    m.speech_base = m.text_len + m.image_n;
    if (m.speech_base + m.speech_n > m.n_vocab) {
        std::snprintf(m.why, sizeof(m.why), "speech range %d..%d runs past the last row %d",
                      m.speech_base, m.speech_base + m.speech_n - 1, m.n_vocab - 1);
        return m;
    }
    m.ok = true;
    return m;
}

inline void print(const Map & m, const char * tag = "VOCAB") {
    std::printf("[%s text_len=%d image=%d..%d speech=%d..%d n_vocab=%d source=first_[PAD<id>]_piece ok=%d why='%s']\n",
                tag, m.text_len, m.image_base, m.image_base + m.image_n - 1,
                m.speech_base, m.speech_base + m.speech_n - 1, m.n_vocab, m.ok ? 1 : 0, m.why);
    std::fflush(stdout);
}

} // namespace dynin_vocab
