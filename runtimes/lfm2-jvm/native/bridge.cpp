#include "cartridge_abi.h"
#include "embedded_descriptor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#include <signal.h>

extern char** environ;

namespace {
constexpr size_t MAX_PAYLOAD = 4 * 1024 * 1024;
constexpr int STARTUP_MS = 120000;
constexpr int IO_MS = 180000;
constexpr int CANCEL_MS = 30000;
constexpr int WRITE_MS = 5000;
constexpr uint64_t CART_MAGIC = 0x4354474341525431ULL;
constexpr uint64_t SESSION_MAGIC = 0x4354475345535331ULL;
constexpr uint8_t RUN = 1, CANCEL = 2, STOP = 3;
constexpr uint8_t LOADED = 16, PARTIAL = 17, FINAL = 18, ERROR = 19, DONE = 20;
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;

struct Handle {
    explicit Handle(uint64_t value = 0) : magic(value) {}
    uint64_t magic;
    std::mutex error_mutex;
    std::array<char, 2048> error{};
};
Handle process_error;
std::mutex preflight_mutex;
std::array<char, 2048> preflight_detail{};

ctg_status fail(Handle* handle, ctg_status status, const char* message) noexcept {
    Handle* target = handle ? handle : &process_error;
    std::lock_guard<std::mutex> lock(target->error_mutex);
    std::snprintf(target->error.data(), target->error.size(), "%s", message ? message : "Unknown failure");
    return status;
}

template<typename F> ctg_status protect(Handle* handle, F action) noexcept {
    try { return action(); }
    catch (const std::bad_alloc&) { return fail(handle, CTG_E_INTERNAL, "Allocation failed"); }
    catch (const std::exception& error) { return fail(handle, CTG_E_INTERNAL, error.what()); }
    catch (...) { return fail(handle, CTG_E_INTERNAL, "Unknown native failure"); }
}

struct TransportError : std::runtime_error { using std::runtime_error::runtime_error; };
struct Child { pid_t pid = -1; int fd = -1; int slot = -1; bool process_group = false; };
struct Frame { uint8_t type; uint64_t id; std::string payload; };
struct OwnedPid { pid_t pid = -1; bool deferred = false; };
std::mutex children_mutex;
std::array<OwnedPid, 256> children{};

int reserve_child_slot() {
    std::lock_guard<std::mutex> lock(children_mutex);
    for (auto& item : children) {
        if (item.pid > 0 && item.deferred) {
            int status = 0;
            const pid_t result = ::waitpid(item.pid, &status, WNOHANG);
            if (result == item.pid || (result < 0 && errno == ECHILD)) item = OwnedPid{};
        }
    }
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i].pid == -1) { children[i].pid = -2; return static_cast<int>(i); }
    }
    throw TransportError("Owned child-process limit reached; deferred children have not exited");
}

void release_child_slot(Child& child) noexcept {
    if (child.slot >= 0) {
        std::lock_guard<std::mutex> lock(children_mutex);
        children[child.slot] = OwnedPid{};
        child.slot = -1;
    }
}

void defer_child(Child& child) noexcept {
    if (child.pid > 0 && child.slot >= 0) {
        std::lock_guard<std::mutex> lock(children_mutex);
        children[child.slot].deferred = true;
        child.pid = -1;
        child.slot = -1;
    }
}

void close_fd(int& fd) noexcept { if (fd >= 0) { ::close(fd); fd = -1; } }

bool reap(pid_t pid, int milliseconds) noexcept {
    const auto deadline = Clock::now() + std::chrono::milliseconds(milliseconds);
    do {
        int status = 0;
        pid_t result = ::waitpid(pid, &status, WNOHANG);
        if (result == pid || (result < 0 && errno == ECHILD)) return true;
        if (result < 0 && errno != EINTR) return true;
        if (Clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (true);
}

bool terminate_child(Child& child, int graceful_ms = 100) noexcept {
    close_fd(child.fd);
    if (child.pid <= 0) { release_child_slot(child); return true; }
    if (child.process_group) {
        const auto graceful_deadline = Clock::now() + std::chrono::milliseconds(graceful_ms);
        while (true) {
            siginfo_t info{};
            const int result = ::waitid(P_PID, static_cast<id_t>(child.pid), &info, WEXITED | WNOHANG | WNOWAIT);
            if (result == 0 && info.si_pid == child.pid) break;
            if (result < 0 && errno != EINTR) {
                // An external reaper has invalidated our process-group ownership.
                // Never signal a potentially recycled ID in that situation.
                if (errno == ECHILD) { child.pid = -1; release_child_slot(child); return true; }
                break;
            }
            if (Clock::now() >= graceful_deadline) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // Keep the leader unreaped until both signals have been sent, so its
        // process-group ID cannot be recycled while cleaning up nested probes.
        ::kill(-child.pid, SIGTERM);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        ::kill(-child.pid, SIGKILL);
        if (!reap(child.pid, 1000)) return false;
        child.pid = -1;
        release_child_slot(child);
        return true;
    }
    if (!reap(child.pid, graceful_ms)) {
        ::kill(child.pid, SIGTERM);
        if (!reap(child.pid, 500)) {
            ::kill(child.pid, SIGKILL);
            // Even SIGKILL cannot promise immediate exit from an uninterruptible
            // kernel wait. Preserve ownership instead of blocking the host forever.
            if (!reap(child.pid, 1000)) return false;
        }
    }
    child.pid = -1;
    release_child_slot(child);
    return true;
}

Child spawn_child(const std::string& executable, const std::vector<std::string>& args, bool capture_stderr,
                  bool process_group = false) {
    // Allocate argv before acquiring OS resources, so allocation failures cannot
    // leak socket descriptors or initialized posix_spawn file actions.
    std::vector<char*> argv;
    argv.reserve(args.size() + 2);
    argv.push_back(const_cast<char*>(executable.c_str()));
    for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
    int sockets[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) throw TransportError("socketpair failed");
    for (int& fd : sockets) {
        if (fd <= STDERR_FILENO) {
            const int promoted = ::fcntl(fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
            if (promoted < 0) { ::close(sockets[0]); ::close(sockets[1]); throw TransportError("Cannot promote worker socket descriptor"); }
            ::close(fd);
            fd = promoted;
        }
    }
    for (int fd : sockets) {
        if (::fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) {
            ::close(sockets[0]); ::close(sockets[1]); throw TransportError("Cannot set close-on-exec");
        }
    }
#ifdef __APPLE__
    int enabled = 1;
    if (::setsockopt(sockets[0], SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) != 0) {
        ::close(sockets[0]); ::close(sockets[1]); throw TransportError("Cannot suppress socket SIGPIPE");
    }
#endif
    posix_spawn_file_actions_t actions;
    if (::posix_spawn_file_actions_init(&actions) != 0) {
        ::close(sockets[0]); ::close(sockets[1]); throw TransportError("spawn actions initialization failed");
    }
    posix_spawnattr_t attributes;
    if (::posix_spawnattr_init(&attributes) != 0) {
        ::posix_spawn_file_actions_destroy(&actions); ::close(sockets[0]); ::close(sockets[1]);
        throw TransportError("spawn attributes initialization failed");
    }
    int error = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
    if (!error && process_group) error = ::posix_spawnattr_setpgroup(&attributes, 0);
    if (!error && process_group) error = ::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    if (!error) error = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
    if (!error && capture_stderr) error = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDERR_FILENO);
    if (!error) error = ::posix_spawn_file_actions_addclose(&actions, sockets[0]);
    if (!error && sockets[1] > STDERR_FILENO) error = ::posix_spawn_file_actions_addclose(&actions, sockets[1]);
    int slot = -1;
    try { if (!error) slot = reserve_child_slot(); }
    catch (...) { ::posix_spawnattr_destroy(&attributes); ::posix_spawn_file_actions_destroy(&actions); ::close(sockets[0]); ::close(sockets[1]); throw; }
    pid_t pid = -1;
    if (!error) error = ::posix_spawn(&pid, executable.c_str(), &actions, &attributes, argv.data(), environ);
    ::posix_spawnattr_destroy(&attributes);
    ::posix_spawn_file_actions_destroy(&actions);
    ::close(sockets[1]);
    Child child{pid, sockets[0], slot, process_group};
    if (error) { close_fd(child.fd); release_child_slot(child); throw TransportError("Cannot launch packaged Java executable"); }
    { std::lock_guard<std::mutex> lock(children_mutex); children[slot].pid = pid; }
    const int flags = ::fcntl(child.fd, F_GETFL);
    if (flags < 0 || ::fcntl(child.fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        if (!terminate_child(child)) defer_child(child);
        throw TransportError("Cannot configure worker socket");
    }
    return child;
}

void wait_socket(int fd, short events, Deadline& deadline, const std::atomic<bool>* cancelled = nullptr,
                 bool* cancellation_observed = nullptr) {
    while (true) {
        const auto now = Clock::now();
        if (cancelled && cancelled->load() && cancellation_observed && !*cancellation_observed) {
            deadline = std::min(deadline, now + std::chrono::milliseconds(CANCEL_MS));
            *cancellation_observed = true;
        }
        if (now >= deadline) throw TransportError("Worker communication deadline exceeded");
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        pollfd item{fd, events, 0};
        int result = ::poll(&item, 1, static_cast<int>(std::min<int64_t>(100, std::max<int64_t>(1, remaining))));
        if (result < 0 && errno == EINTR) continue;
        if (result < 0 || (result > 0 && (item.revents & (POLLERR | POLLNVAL)))) throw TransportError("Worker socket failed");
        if (result > 0 && (item.revents & (events | POLLHUP))) return;
    }
}

void transfer(int fd, void* buffer, size_t length, bool writing, Deadline& deadline,
              const std::atomic<bool>* cancelled = nullptr, bool* observed = nullptr) {
    auto* bytes = static_cast<unsigned char*>(buffer);
    size_t offset = 0;
    bool local_observed = false;
    bool* cancellation_observed = observed ? observed : &local_observed;
    while (offset < length) {
        wait_socket(fd, writing ? POLLOUT : POLLIN, deadline, cancelled, cancellation_observed);
        ssize_t count;
        if (writing) {
#ifdef MSG_NOSIGNAL
            count = ::send(fd, bytes + offset, length - offset, MSG_NOSIGNAL);
#else
            count = ::send(fd, bytes + offset, length - offset, 0);
#endif
        } else count = ::recv(fd, bytes + offset, length - offset, 0);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (count <= 0) throw TransportError(writing ? "Worker command write failed" : "Worker exited or truncated a frame");
        offset += static_cast<size_t>(count);
    }
}

void send_frame(int fd, uint8_t type, uint64_t id, const std::string& payload) {
    if (payload.size() > MAX_PAYLOAD) throw TransportError("Command exceeds frame limit");
    unsigned char header[13]{};
    header[0] = type;
    for (int i = 0; i < 8; ++i) header[1 + i] = static_cast<unsigned char>(id >> (56 - 8 * i));
    for (int i = 0; i < 4; ++i) header[9 + i] = static_cast<unsigned char>(payload.size() >> (24 - 8 * i));
    auto deadline = Clock::now() + std::chrono::milliseconds(WRITE_MS);
    transfer(fd, header, sizeof(header), true, deadline);
    if (!payload.empty()) transfer(fd, const_cast<char*>(payload.data()), payload.size(), true, deadline);
}

Frame receive_frame(int fd, int timeout_ms, const std::atomic<bool>* cancelled = nullptr) {
    unsigned char header[13];
    auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    bool cancellation_observed = false;
    transfer(fd, header, sizeof(header), false, deadline, cancelled, &cancellation_observed);
    uint64_t id = 0;
    uint32_t length = 0;
    for (int i = 0; i < 8; ++i) id = (id << 8) | header[1 + i];
    for (int i = 0; i < 4; ++i) length = (length << 8) | header[9 + i];
    if (length > MAX_PAYLOAD) throw TransportError("Worker frame exceeds 4 MiB limit");
    Frame frame{header[0], id, std::string(length, '\0')};
    if (length) transfer(fd, &frame.payload[0], length, false, deadline, cancelled, &cancellation_observed);
    return frame;
}

std::pair<ctg_status, std::string> frame_error(const Frame& frame) {
    if (frame.payload.size() < 4) throw TransportError("Malformed worker ERROR frame");
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) value = (value << 8) | static_cast<unsigned char>(frame.payload[i]);
    const int32_t code = static_cast<int32_t>(value);
    if (!((code <= -1 && code >= -8) || code == -99)) throw TransportError("Unknown worker error status");
    std::string detail = frame.payload.substr(4);
    if (detail.find('\0') != std::string::npos) throw TransportError("Worker error contains NUL");
    return {static_cast<ctg_status>(code), detail};
}

std::string absolute_pack(const char* path) {
    if (!path || !*path) throw std::invalid_argument("pack_dir is required");
    char resolved[PATH_MAX];
    if (!::realpath(path, resolved)) throw std::invalid_argument("pack_dir does not resolve");
    struct stat info{};
    if (::stat(resolved, &info) != 0 || !S_ISDIR(info.st_mode)) throw std::invalid_argument("pack_dir is not a directory");
    return resolved;
}

std::string file_bytes(const std::string& path, size_t limit) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::invalid_argument("Required package file is unreadable");
    std::string result;
    std::array<char, 8192> buffer{};
    while (stream) {
        stream.read(buffer.data(), buffer.size());
        const auto count = stream.gcount();
        if (count < 0 || result.size() + static_cast<size_t>(count) > limit) throw std::invalid_argument("Package file exceeds limit");
        result.append(buffer.data(), static_cast<size_t>(count));
    }
    if (!stream.eof()) throw std::invalid_argument("Cannot read package file");
    return result;
}

void check_package(const std::string& pack) {
    if (file_bytes(pack + "/descriptor.json", MAX_PAYLOAD) != reinterpret_cast<const char*>(kPackDescriptorBytes))
        throw std::invalid_argument("Package descriptor differs from the exact embedded pack descriptor");
    (void)file_bytes(pack + "/artifacts/runtime/worker.json", MAX_PAYLOAD);
    struct stat info{};
    if (::stat((pack + "/artifacts/jvm/lib").c_str(), &info) != 0 || !S_ISDIR(info.st_mode))
        throw std::invalid_argument("Packaged JVM library directory is unavailable");
}

std::string resolve_java(const std::string& pack) {
    std::string executable = pack + "/artifacts/runtime/java";
    struct stat info{};
    if (::lstat(executable.c_str(), &info) != 0) {
        if (errno != ENOENT) throw std::invalid_argument("Cannot inspect packaged Java executable");
        const char* home = std::getenv("JAVA_HOME");
        if (!home || !*home) throw std::invalid_argument("Java executable is absent from the pack; JAVA_HOME is required");
        executable = std::string(home) + "/bin/java";
    }
    char resolved[PATH_MAX];
    if (!::realpath(executable.c_str(), resolved) || ::stat(resolved, &info) != 0 ||
        !S_ISREG(info.st_mode) || ::access(resolved, X_OK) != 0)
        throw std::invalid_argument("Selected Java executable is unavailable");
    return resolved;
}

struct ProbeResult { std::string output; int status; };

ProbeResult run_probe(const std::string& executable, const std::vector<std::string>& arguments,
                      bool capture_stderr, int milliseconds) {
    Child child = spawn_child(executable, arguments, capture_stderr, true);
    try {
        std::string output;
        auto deadline = Clock::now() + std::chrono::milliseconds(milliseconds);
        while (true) {
            wait_socket(child.fd, POLLIN, deadline);
            char buffer[4096];
            const ssize_t count = ::recv(child.fd, buffer, sizeof(buffer), 0);
            if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
            if (count < 0) throw TransportError("Preflight child read failed");
            if (count == 0) break;
            output.append(buffer, static_cast<size_t>(count));
            if (output.size() > 65536) throw TransportError("Preflight child output exceeds limit");
        }
        close_fd(child.fd);
        int status = 0;
        while (true) {
            const pid_t result = ::waitpid(child.pid, &status, WNOHANG);
            if (result == child.pid) { child.pid = -1; release_child_slot(child); break; }
            if (result < 0 && errno != EINTR) throw TransportError("Cannot reap preflight child");
            if (Clock::now() >= deadline) throw TransportError("Preflight child exit deadline exceeded");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return {std::move(output), status};
    } catch (...) { if (!terminate_child(child)) defer_child(child); throw; }
}

void check_java(const std::string& executable) {
    const auto result = run_probe(executable,
        {"--enable-preview", "--add-modules", "jdk.incubator.vector", "-version"}, true, 15000);
    const bool java21 = result.output.find("version \"21.") != std::string::npos ||
                        result.output.find("version \"21\"") != std::string::npos ||
                        result.output.find("version \"21-") != std::string::npos;
    if (!WIFEXITED(result.status) || WEXITSTATUS(result.status) != 0 || !java21)
        throw std::invalid_argument("Java 21 with jdk.incubator.vector and preview support is required");
}

// Small bounded JSON reader for the backend readiness reply. Only root-level
// string fields status/backend authorize readiness; nested markers cannot do so.
class PreflightReply {
    const std::string& text;
    size_t at = 0;
    std::string status, backend;
    bool model_unloaded = false;
    [[noreturn]] void invalid() const { throw std::invalid_argument("Worker preflight returned invalid readiness metadata"); }
    void whitespace() { while (at < text.size() && (text[at] == ' ' || text[at] == '\n' || text[at] == '\r' || text[at] == '\t')) ++at; }
    char peek() { whitespace(); return at < text.size() ? text[at] : '\0'; }
    void take(char expected) { if (peek() != expected) invalid(); ++at; }
    static bool digit(char c) { return c >= '0' && c <= '9'; }
    std::string string() {
        take('"');
        std::string value;
        while (at < text.size()) {
            const auto c = static_cast<unsigned char>(text[at++]);
            if (c == '"') return value;
            if (c < 32) invalid();
            if (c != '\\') { value += static_cast<char>(c); continue; }
            if (at == text.size()) invalid();
            const char escape = text[at++];
            if (escape == 'u') {
                if (text.size() - at < 4) invalid();
                unsigned value16 = 0;
                for (int i = 0; i < 4; ++i) {
                    const char h = text[at + i];
                    const int nibble = digit(h) ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
                    if (nibble < 0) invalid();
                    value16 = value16 * 16 + static_cast<unsigned>(nibble);
                }
                // The two recognized protocol fields contain only ASCII.
                if (value16 <= 127) value += static_cast<char>(value16);
                else { value += "\\u"; value.append(text, at, 4); }
                at += 4;
            } else {
                switch (escape) {
                    case '"': case '\\': case '/': value += escape; break;
                    case 'b': value += '\b'; break; case 'f': value += '\f'; break;
                    case 'n': value += '\n'; break; case 'r': value += '\r'; break; case 't': value += '\t'; break;
                    default: invalid();
                }
            }
        }
        invalid();
    }
    void value(int depth) {
        if (depth > 32) invalid();
        const char start = peek();
        if (start == '{') { object(depth); return; }
        if (start == '[') {
            ++at;
            if (peek() == ']') { ++at; return; }
            while (true) { value(depth + 1); if (peek() == ']') { ++at; return; } take(','); }
        }
        if (start == '"') { (void)string(); return; }
        for (const char* literal : {"true", "false", "null"}) {
            const size_t size = std::strlen(literal);
            if (text.compare(at, size, literal) == 0) { at += size; return; }
        }
        if (start == '-') ++at;
        if (at >= text.size()) invalid();
        if (text[at] == '0') ++at;
        else {
            if (text[at] < '1' || text[at] > '9') invalid();
            while (at < text.size() && digit(text[at])) ++at;
        }
        if (at < text.size() && text[at] == '.') {
            ++at; if (at == text.size() || !digit(text[at])) invalid();
            while (at < text.size() && digit(text[at])) ++at;
        }
        if (at < text.size() && (text[at] == 'e' || text[at] == 'E')) {
            ++at;
            if (at < text.size() && (text[at] == '+' || text[at] == '-')) ++at;
            if (at == text.size() || !digit(text[at])) invalid();
            while (at < text.size() && digit(text[at])) ++at;
        }
    }
    void object(int depth) {
        take('{');
        std::unordered_set<std::string> names;
        if (peek() == '}') { ++at; return; }
        while (true) {
            const auto name = string();
            if (!names.insert(name).second) invalid();
            take(':');
            if (depth == 0 && (name == "status" || name == "backend")) {
                const auto item = string();
                (name == "status" ? status : backend) = item;
            } else if (depth == 0 && name == "model_loaded") {
                whitespace();
                if (text.compare(at, 5, "false") != 0) invalid();
                at += 5;
                model_unloaded = true;
            } else value(depth + 1);
            if (peek() == '}') { ++at; return; }
            take(',');
        }
    }
public:
    explicit PreflightReply(const std::string& data) : text(data) {}
    void verify() {
        object(0); whitespace();
        if (at != text.size() || status != "ready" || !model_unloaded || (backend != "eager-cpu" && backend != "iree-vulkan")) invalid();
    }
};

void check_backend(const std::string& executable, const std::string& pack) {
    const auto result = run_probe(executable,
        {"--enable-preview", "--add-modules", "jdk.incubator.vector", "-Xmx256m", "-cp",
         pack + "/artifacts/jvm/lib/*", "sk.ainet.cartridge.generator.lfm2.worker.MainKt", "--preflight", "--pack-dir", pack},
        false, 30000);
    if (!WIFEXITED(result.status) || WEXITSTATUS(result.status) != 0)
        throw std::invalid_argument("Packaged backend preflight failed; see worker stderr");
    PreflightReply(result.output).verify();
}

std::string prepare_java(const std::string& pack) {
    check_package(pack);
    auto executable = resolve_java(pack);
    check_java(executable);
    check_backend(executable, pack);
    return executable;
}
}

struct ctg_session;
struct ctg_cartridge {
    Handle handle{CART_MAGIC};
    Child child;
    std::mutex lifecycle;
    std::mutex writing;
    std::unordered_set<ctg_session*> sessions;
    ctg_session* active = nullptr;
    uint64_t next_id = 1;
    std::atomic<bool> broken{false};
    bool closing = false;
};
enum class SessionState { INPUT, RUNNING, FINAL_SEEN, DRAINED, FAILED };
struct ctg_session {
    explicit ctg_session(ctg_cartridge* owner, uint64_t value) : cartridge(owner), id(value) {}
    Handle handle{SESSION_MAGIC};
    ctg_cartridge* cartridge;
    uint64_t id;
    std::string input;
    std::atomic<SessionState> state{SessionState::INPUT};
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};
    std::atomic<bool> done{false};
    std::atomic<bool> cancel_sent{false};
};
struct ctg_result {
    ctg_result_kind kind;
    std::string payload;
};

namespace {
bool valid(const ctg_cartridge* handle) { return handle && handle->handle.magic == CART_MAGIC; }
bool valid(const ctg_session* handle) { return handle && handle->handle.magic == SESSION_MAGIC; }

void finish_request(ctg_session* session) {
    session->done.store(true);
    std::lock_guard<std::mutex> lock(session->cartridge->lifecycle);
    if (session->cartridge->active == session) session->cartridge->active = nullptr;
}

void break_worker(ctg_session* session) noexcept {
    auto* cartridge = session->cartridge;
    cartridge->broken.store(true);
    // Cancellation writes and descriptor teardown must not race on the socket.
    { std::lock_guard<std::mutex> lock(cartridge->writing); terminate_child(cartridge->child); }
    session->state.store(SessionState::FAILED);
    finish_request(session);
}

void cancel_command(ctg_session* session) {
    auto* cartridge = session->cartridge;
    std::lock_guard<std::mutex> lock(cartridge->writing);
    if (session->started.load() && !session->done.load() && !cartridge->broken.load() && !session->cancel_sent.exchange(true))
        send_frame(cartridge->child.fd, CANCEL, session->id, {});
}

Frame next_frame(ctg_session* session, int timeout = IO_MS) {
    Frame frame = receive_frame(session->cartridge->child.fd, timeout, &session->cancelled);
    if (frame.id != session->id) throw TransportError("Worker response has the wrong request ID");
    return frame;
}

void consume_done(ctg_session* session) {
    const Frame done = next_frame(session, CANCEL_MS);
    if (done.type != DONE || !done.payload.empty()) throw TransportError("Expected empty DONE after terminal response");
    finish_request(session);
}

ctg_status unavailable(ctg_session* session) {
    if (!valid(session)) return fail(nullptr, CTG_E_INVALID_ARG, "Invalid session handle");
    if (session->cancelled.load()) return fail(&session->handle, CTG_E_CANCELLED, "Session is cancelled; only session_end is allowed");
    if (session->cartridge->broken.load()) return fail(&session->handle, CTG_E_BACKEND, "Worker is unavailable");
    return CTG_OK;
}
}

extern "C" uint32_t ctg_abi_version(void) { return (CTG_ABI_MAJOR << 16) | CTG_ABI_MINOR; }
extern "C" const char* ctg_descriptor_json(void) { return reinterpret_cast<const char*>(kDescriptorBytes); }

extern "C" ctg_status ctg_preflight(const char* pack_dir, ctg_preflight_report* report) {
    return protect(nullptr, [&]() -> ctg_status {
        if (!report) return fail(nullptr, CTG_E_INVALID_ARG, "Preflight report is required");
        if (report->size < sizeof(*report)) return fail(nullptr, CTG_E_ABI_MISMATCH, "Preflight report is too small");
        std::lock_guard<std::mutex> lock(preflight_mutex);
        report->ok = 0;
        report->detail = preflight_detail.data();
        try {
            const auto pack = absolute_pack(pack_dir);
            (void)prepare_java(pack);
            std::snprintf(preflight_detail.data(), preflight_detail.size(), "%s", "Java 21/vector and packaged backend requirements ready; no model loaded");
            report->ok = 1;
            return CTG_OK;
        } catch (const std::exception& error) {
            std::snprintf(preflight_detail.data(), preflight_detail.size(), "%s", error.what());
            return fail(nullptr, CTG_E_PREFLIGHT, error.what());
        }
    });
}

extern "C" ctg_status ctg_open(const char* pack_dir, const ctg_open_options* opts, ctg_cartridge** out) {
    return protect(nullptr, [&]() -> ctg_status {
        if (!out) return fail(nullptr, CTG_E_INVALID_ARG, "Output cartridge pointer is required");
        *out = nullptr;
        if (opts && opts->size < sizeof(*opts)) return fail(nullptr, CTG_E_ABI_MISMATCH, "Open options are too small");
        if (opts && opts->flags) return fail(nullptr, CTG_E_INVALID_ARG, "Unknown open flags");
        std::string pack, java;
        try {
            pack = absolute_pack(pack_dir);
            java = prepare_java(pack);
        } catch (const std::exception& error) {
            return fail(nullptr, CTG_E_PREFLIGHT, error.what());
        }
        auto cartridge = std::make_unique<ctg_cartridge>();
        try {
            cartridge->child = spawn_child(java,
                {"--enable-preview", "--add-modules", "jdk.incubator.vector", "-Xmx6g", "-cp",
                 pack + "/artifacts/jvm/lib/*", "sk.ainet.cartridge.generator.lfm2.worker.MainKt", "--pack-dir", pack}, false, true);
            const Frame loaded = receive_frame(cartridge->child.fd, STARTUP_MS);
            if (loaded.id != 0) throw TransportError("Startup response requires request ID zero");
            if (loaded.type == ERROR) {
                const auto error = frame_error(loaded);
                if (!terminate_child(cartridge->child)) defer_child(cartridge->child);
                return fail(nullptr, error.first, error.second.c_str());
            }
            if (loaded.type != LOADED || loaded.payload.empty()) throw TransportError("Worker did not send LOADED metadata");
            *out = cartridge.release();
            return CTG_OK;
        } catch (const std::exception& error) {
            if (!terminate_child(cartridge->child)) defer_child(cartridge->child);
            return fail(nullptr, CTG_E_LOAD, error.what());
        }
    });
}

extern "C" ctg_status ctg_close(ctg_cartridge* cartridge) {
    if (!valid(cartridge)) return fail(nullptr, CTG_E_INVALID_ARG, "Invalid cartridge handle");
    return protect(&cartridge->handle, [&]() -> ctg_status {
        {
            std::lock_guard<std::mutex> lock(cartridge->lifecycle);
            if (!cartridge->sessions.empty()) return fail(&cartridge->handle, CTG_E_STATE, "End all sessions before closing the cartridge");
            cartridge->closing = true;
        }
        {
            std::lock_guard<std::mutex> lock(cartridge->writing);
            try { if (!cartridge->broken.load() && cartridge->child.fd >= 0) send_frame(cartridge->child.fd, STOP, 0, {}); }
            catch (...) { /* Teardown still owns and reaps the worker. */ }
            if (!terminate_child(cartridge->child, 2000))
                return fail(&cartridge->handle, CTG_E_BACKEND, "Worker exit exceeded cleanup deadline; cartridge remains owned, retry close");
        }
        cartridge->handle.magic = 0;
        delete cartridge;
        return CTG_OK;
    });
}

extern "C" ctg_status ctg_session_begin(ctg_cartridge* cartridge, const ctg_session_options* opts, ctg_session** out) {
    if (!valid(cartridge)) return fail(nullptr, CTG_E_INVALID_ARG, "Invalid cartridge handle");
    return protect(&cartridge->handle, [&]() -> ctg_status {
        if (!out) return fail(&cartridge->handle, CTG_E_INVALID_ARG, "Output session pointer is required");
        *out = nullptr;
        if (opts && opts->size < sizeof(*opts)) return fail(&cartridge->handle, CTG_E_ABI_MISMATCH, "Session options are too small");
        if (opts && opts->flags) return fail(&cartridge->handle, CTG_E_INVALID_ARG, "Unknown session flags");
        std::lock_guard<std::mutex> lock(cartridge->lifecycle);
        if (cartridge->closing || cartridge->broken.load()) return fail(&cartridge->handle, CTG_E_STATE, "Cartridge is unavailable");
        if (cartridge->next_id == 0 || cartridge->next_id > INT64_MAX) return fail(&cartridge->handle, CTG_E_STATE, "Session IDs exhausted");
        auto session = std::make_unique<ctg_session>(cartridge, cartridge->next_id++);
        cartridge->sessions.insert(session.get());
        *out = session.release();
        return CTG_OK;
    });
}

extern "C" ctg_status ctg_push(ctg_session* session, const ctg_buffer* chunk) {
    const auto available = unavailable(session);
    if (available != CTG_OK) return available;
    return protect(&session->handle, [&]() -> ctg_status {
        if (!chunk) return fail(&session->handle, CTG_E_INVALID_ARG, "Input buffer is required");
        if (chunk->size < sizeof(*chunk)) return fail(&session->handle, CTG_E_ABI_MISMATCH, "Input buffer is too small");
        if (session->state.load() != SessionState::INPUT) return fail(&session->handle, CTG_E_STATE, "Input is already flushed");
        if ((!chunk->data && chunk->len) || chunk->len > MAX_PAYLOAD || session->input.size() > MAX_PAYLOAD - chunk->len)
            return fail(&session->handle, CTG_E_INVALID_ARG, "Input is null or exceeds the 4 MiB request limit");
        if (chunk->len) session->input.append(static_cast<const char*>(chunk->data), chunk->len);
        if (session->cancelled.load()) return fail(&session->handle, CTG_E_CANCELLED, "Session cancelled during push");
        return CTG_OK;
    });
}

extern "C" ctg_status ctg_flush(ctg_session* session) {
    const auto available = unavailable(session);
    if (available != CTG_OK) return available;
    return protect(&session->handle, [&]() -> ctg_status {
        if (session->state.load() != SessionState::INPUT) return fail(&session->handle, CTG_E_STATE, "Input is already flushed");
        if (session->input.empty()) return fail(&session->handle, CTG_E_INVALID_ARG, "A JSON request is required before flush");
        auto* cartridge = session->cartridge;
        {
            std::lock_guard<std::mutex> lock(cartridge->lifecycle);
            if (cartridge->active) return fail(&session->handle, CTG_E_STATE, "Only one generation may be active per cartridge");
            cartridge->active = session;
        }
        try {
            {
                std::lock_guard<std::mutex> lock(cartridge->writing);
                if (session->cancelled.load()) { finish_request(session); return fail(&session->handle, CTG_E_CANCELLED, "Session cancelled before RUN"); }
                send_frame(cartridge->child.fd, RUN, session->id, session->input);
                session->started.store(true);
                session->state.store(SessionState::RUNNING);
            }
            session->input.clear();
            if (session->cancelled.load()) { cancel_command(session); return fail(&session->handle, CTG_E_CANCELLED, "Session cancelled during flush"); }
            return CTG_OK;
        } catch (const std::exception& error) {
            break_worker(session);
            return fail(&session->handle, CTG_E_BACKEND, error.what());
        }
    });
}

extern "C" ctg_status ctg_pull(ctg_session* session, ctg_result** out) {
    if (out) *out = nullptr;
    const auto available = unavailable(session);
    if (available != CTG_OK) return available;
    return protect(&session->handle, [&]() -> ctg_status {
        if (!out) return fail(&session->handle, CTG_E_INVALID_ARG, "Output result pointer is required");
        const auto state = session->state.load();
        if (state == SessionState::INPUT || state == SessionState::DRAINED) return CTG_OK;
        if (state == SessionState::FAILED) return fail(&session->handle, CTG_E_STATE, "Session has failed; end the session");
        try {
            Frame frame = next_frame(session);
            if (frame.type == ERROR) {
                const auto error = frame_error(frame);
                consume_done(session);
                session->state.store(SessionState::FAILED);
                if (error.first == CTG_E_CANCELLED) session->cancelled.store(true);
                return fail(&session->handle, session->cancelled.load() ? CTG_E_CANCELLED : error.first, error.second.c_str());
            }
            if (frame.type == DONE) {
                if (!frame.payload.empty() || state != SessionState::FINAL_SEEN) throw TransportError("DONE without a terminal result");
                finish_request(session);
                session->state.store(SessionState::DRAINED);
                if (session->cancelled.load()) return fail(&session->handle, CTG_E_CANCELLED, "Session cancelled");
                return CTG_OK;
            }
            if (frame.type != PARTIAL && frame.type != FINAL) throw TransportError("Unexpected worker result type");
            if (state == SessionState::FINAL_SEEN) throw TransportError("Worker emitted after FINAL");
            if (frame.payload.empty()) throw TransportError("Worker result payload is empty");
            if (session->cancelled.load()) return fail(&session->handle, CTG_E_CANCELLED, "Session cancelled");
            auto result = std::make_unique<ctg_result>();
            result->kind = frame.type == FINAL ? CTG_RESULT_FINAL : CTG_RESULT_PARTIAL;
            result->payload = std::move(frame.payload);
            if (frame.type == FINAL) session->state.store(SessionState::FINAL_SEEN);
            *out = result.release();
            return CTG_OK;
        } catch (const std::exception& error) {
            break_worker(session);
            return fail(&session->handle, session->cancelled.load() ? CTG_E_CANCELLED : CTG_E_BACKEND, error.what());
        }
    });
}

extern "C" ctg_status ctg_cancel(ctg_session* session) {
    if (!valid(session)) return fail(nullptr, CTG_E_INVALID_ARG, "Invalid session handle");
    return protect(&session->handle, [&]() -> ctg_status {
        if (session->cancelled.exchange(true)) return fail(&session->handle, CTG_E_CANCELLED, "Session is already cancelled");
        try { cancel_command(session); return CTG_OK; }
        catch (const std::exception& error) { return fail(&session->handle, CTG_E_BACKEND, error.what()); }
    });
}

extern "C" ctg_status ctg_session_end(ctg_session* session) {
    if (!valid(session)) return fail(nullptr, CTG_E_INVALID_ARG, "Invalid session handle");
    return protect(&session->handle, [&]() -> ctg_status {
        ctg_status result = CTG_OK;
        auto* cartridge = session->cartridge;
        if (session->started.load() && !session->done.load() && !cartridge->broken.load()) {
            session->cancelled.store(true);
            try {
                cancel_command(session);
                const auto deadline = Clock::now() + std::chrono::milliseconds(CANCEL_MS);
                while (!session->done.load()) {
                    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
                    if (remaining <= 0) throw TransportError("Session teardown deadline exceeded");
                    const Frame frame = next_frame(session, static_cast<int>(remaining));
                    if (frame.type == DONE) {
                        if (!frame.payload.empty()) throw TransportError("DONE payload must be empty");
                        finish_request(session);
                    } else if (frame.type == ERROR) { (void)frame_error(frame); }
                    else if (frame.type != PARTIAL && frame.type != FINAL) throw TransportError("Unexpected response during teardown");
                }
            } catch (const std::exception& error) {
                break_worker(session);
                result = fail(nullptr, CTG_E_BACKEND, error.what());
            }
        }
        {
            std::lock_guard<std::mutex> lock(cartridge->lifecycle);
            if (cartridge->active == session) cartridge->active = nullptr;
            cartridge->sessions.erase(session);
        }
        session->handle.magic = 0;
        delete session;
        return result;
    });
}

extern "C" ctg_status ctg_infer(ctg_session* session, const ctg_buffer*, ctg_result** out) {
    if (out) *out = nullptr;
    const auto available = unavailable(session);
    if (available != CTG_OK) return available;
    return fail(&session->handle, CTG_E_UNSUPPORTED, "This cartridge uses streaming push/flush/pull");
}

extern "C" ctg_result_kind ctg_result_get_kind(const ctg_result* result) {
    if (!result) { fail(nullptr, CTG_E_INVALID_ARG, "Result is required"); return static_cast<ctg_result_kind>(0); }
    return result->kind;
}

extern "C" ctg_status ctg_result_get_payload(const ctg_result* result, ctg_buffer* out_view) {
    if (!result || !out_view) return fail(nullptr, CTG_E_INVALID_ARG, "Result and output view are required");
    if (out_view->size < sizeof(*out_view)) return fail(nullptr, CTG_E_ABI_MISMATCH, "Output view is too small");
    out_view->data = result->payload.data();
    out_view->len = result->payload.size();
    return CTG_OK;
}

extern "C" void ctg_result_release(ctg_result* result) { delete result; }

extern "C" const char* ctg_last_error(const void* handle) {
    auto* target = handle ? const_cast<Handle*>(static_cast<const Handle*>(handle)) : &process_error;
    std::lock_guard<std::mutex> lock(target->error_mutex);
    return target->error.data();
}
