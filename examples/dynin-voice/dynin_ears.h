// dynin_ears.h - a thin scoring-oracle wrapper: native speech-to-text via a
// child whisper-cli process, used only to check synthesized speech against
// the text it was generated from.
//
// This does NOT link whisper/ggml/llama. It shells whisper-cli out as a
// child process (a raw pipe on its stdout, no shell involvement on Windows;
// popen()/pclose() through the platform shell on POSIX) so this target's
// dependency surface stays at zero and it can never desync from a
// whisper.cpp built/rebuilt independently. whisper.cpp's own miniaudio-based
// WAV reader (examples/common-whisper.cpp, ma_decoder_config_init(...,
// WHISPER_SAMPLE_RATE)) resamples on the fly, so this helper passes the
// source .wav straight through -- no 16 kHz pre-conversion step, no ffmpeg
// call, no resampler tool needed or built.
//
// Timestamps: whisper-cli's default stdout is one line per segment,
// "[hh:mm:ss.mmm --> hh:mm:ss.mmm]  text" (examples/cli/cli.cpp's
// print_output loop). strip_timestamps() below parses that bracket off
// itself (rather than relying on passing -nt), so this helper's output is
// correct against whisper-cli's default invocation.
//
// Header-only, C++17.
#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <cstdio>
#endif

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace dynin_ears {

struct DyninEars {
    std::string exe;    // path to whisper-cli.exe
    std::string model;  // path to a ggml *.bin whisper model

    // Runs `exe -m model -f wav_path -l en`, waits for it to exit, and
    // returns its transcript: timestamps stripped, segment lines joined with
    // a single space, leading/trailing whitespace trimmed. *ms_out (if
    // non-null) receives the child process's wall-clock milliseconds
    // (process spawn to exit), regardless of outcome. Returns "" if the pipe
    // or the process could not be created; never throws.
    [[nodiscard]] std::string transcribe(const std::string & wav_path, int * ms_out) const;

private:
    static std::string strip_timestamps(const std::string & raw);
};

// ---------------------------------------------------------------- impl

inline std::string DyninEars::strip_timestamps(const std::string & raw) {
    std::string out;
    out.reserve(raw.size());

    size_t pos = 0;
    while (pos < raw.size()) {
        size_t eol = raw.find('\n', pos);
        if (eol == std::string::npos) eol = raw.size();
        std::string line = raw.substr(pos, eol - pos);
        pos = eol + 1;

        // one leading "[hh:mm:ss.mmm --> hh:mm:ss.mmm]" bracket, plus the
        // whitespace cli.cpp prints after it -- strip both if present.
        if (!line.empty() && line.front() == '[') {
            size_t close = line.find(']');
            if (close != std::string::npos) {
                size_t start = close + 1;
                while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
                    start++;
                }
                line = line.substr(start);
            }
        }

        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }

        if (line.empty()) continue;
        if (!out.empty()) out += ' ';
        out += line;
    }

    const size_t b = out.find_first_not_of(" \t");
    if (b == std::string::npos) return {};
    const size_t e = out.find_last_not_of(" \t");
    return out.substr(b, e - b + 1);
}

#ifdef _WIN32
inline std::string DyninEars::transcribe(const std::string & wav_path, int * ms_out) const {
    if (ms_out) *ms_out = 0;

    SECURITY_ATTRIBUTES sa{};
    sa.nLength              = sizeof(sa);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE child_out_rd = nullptr;
    HANDLE child_out_wr = nullptr;
    if (!CreatePipe(&child_out_rd, &child_out_wr, &sa, 0)) {
        return {};
    }
    // the parent's read end must not be inherited by the child
    if (!SetHandleInformation(child_out_rd, HANDLE_FLAG_INHERIT, 0)) {
        CloseHandle(child_out_rd);
        CloseHandle(child_out_wr);
        return {};
    }

    const std::wstring wexe   (exe.begin(), exe.end());
    const std::wstring wmodel (model.begin(), model.end());
    const std::wstring wwav   (wav_path.begin(), wav_path.end());

    // CreateProcessW's lpCommandLine must be a mutable buffer
    std::wstring cmd = L"\"" + wexe + L"\" -m \"" + wmodel + L"\" -f \"" + wwav + L"\" -l en";
    std::vector<wchar_t> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back(L'\0');

    STARTUPINFOW si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = child_out_wr;
    // whisper-cli prints ALL its diagnostics (model load, CUDA init, timing)
    // to stderr and ONLY the "[ts --> ts]  text" segment lines to stdout
    // (examples/cli/cli.cpp) -- so stderr must NOT share the captured pipe,
    // or strip_timestamps() below would fold model-load noise into the
    // transcript (caught empirically: first dynin-ears-test run did exactly
    // this before the fix). Let it flow through to the parent's own stderr.
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};

    const auto t_start = std::chrono::steady_clock::now();

    const BOOL ok = CreateProcessW(
        wexe.c_str(),
        cmd_buf.data(),
        nullptr, nullptr,
        /*bInheritHandles=*/ TRUE,
        CREATE_NO_WINDOW,
        nullptr, nullptr,
        &si, &pi);

    // parent's copy of the write handle must close so ReadFile below sees
    // EOF once the child (the only other handle owner) exits
    CloseHandle(child_out_wr);

    if (!ok) {
        CloseHandle(child_out_rd);
        return {};
    }

    std::string raw;
    char buf[4096];
    DWORD n_read = 0;
    while (ReadFile(child_out_rd, buf, sizeof(buf), &n_read, nullptr) && n_read > 0) {
        raw.append(buf, n_read);
    }
    CloseHandle(child_out_rd);

    WaitForSingleObject(pi.hProcess, INFINITE);

    if (ms_out) {
        const auto t_end = std::chrono::steady_clock::now();
        *ms_out = (int) std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return strip_timestamps(raw);
}
#else  // POSIX
namespace detail {
// Wraps a path in single quotes for a POSIX shell, escaping any embedded
// single quote as '\'' (close quote, literal quote, reopen quote).
inline std::string shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}
}  // namespace detail

inline std::string DyninEars::transcribe(const std::string & wav_path, int * ms_out) const {
    if (ms_out) *ms_out = 0;

    // popen() only hands back one stream (the child's stdout); its stderr is
    // inherited from this process unredirected, same as the Windows branch's
    // si.hStdError passthrough above -- so whisper-cli's diagnostics still
    // land on the parent's own stderr and never fold into the transcript.
    // Neither branch has a timeout: popen()/pclose() here, like the Windows
    // branch's ReadFile loop and WaitForSingleObject(INFINITE), blocks until
    // whisper-cli exits, so a hung whisper-cli blocks this call
    // indefinitely. That tradeoff is accepted for a test-only oracle path
    // rather than adding a watchdog.
    const std::string cmd = detail::shell_quote(exe) + " -m " + detail::shell_quote(model) +
                             " -f " + detail::shell_quote(wav_path) + " -l en";

    const auto t_start = std::chrono::steady_clock::now();

    FILE * pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return {};
    }

    std::string raw;
    char buf[4096];
    size_t n_read = 0;
    while ((n_read = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        raw.append(buf, n_read);
    }

    pclose(pipe);

    if (ms_out) {
        const auto t_end = std::chrono::steady_clock::now();
        *ms_out = (int) std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    }

    return strip_timestamps(raw);
}
#endif  // _WIN32

}  // namespace dynin_ears
