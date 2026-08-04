#include "server.h"

#include "embedded_ui.h"
#include "wire.h"

// cpp-httplib pulls in the platform socket headers; keep it after ours.
#include <httplib.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace Viz {
namespace {

// Shown when the binary was built without a UI bundle (the engine-side stream
// still works, which is what the headless smoke test exercises).
constexpr const char *PLACEHOLDER_HTML =
    "<!doctype html><html><head><meta charset=\"utf-8\">"
    "<title>STK-Vector-64 Vector Scope</title>"
    "<style>body{background:#0a0c10;color:#e6edf3;font:14px ui-monospace,"
    "monospace;margin:0;display:grid;place-items:center;height:100vh}"
    "code{color:#6ee7b7}</style></head><body><div>"
    "<h2>Vector Scope</h2>"
    "<p>No UI bundle embedded in this build.</p>"
    "<p>The engine stream is live: <code>GET /api/state?since=0</code></p>"
    "</div></body></html>";

void try_open_browser(const std::string &url) {
#if defined(_WIN32)
  const std::string cmd = "start \"\" \"" + url + "\"";
#elif defined(__APPLE__)
  const std::string cmd = "open \"" + url + "\"";
#else
  const std::string cmd = "xdg-open \"" + url + "\" >/dev/null 2>&1 &";
#endif
  // Best effort only: the URL is composed from our own host/port, never input.
  if (std::system(cmd.c_str()) != 0)
    std::fprintf(stderr, "viz: could not open a browser; visit %s\n",
                 url.c_str());
}

} // namespace

int run_server(Session &session, const ServerOptions &opts) {
  httplib::Server svr;

  // This tool exposes engine control with no authentication. Refuse anything
  // that is not loopback rather than silently listening to the network.
  if (opts.host != "127.0.0.1" && opts.host != "localhost" &&
      opts.host != "::1") {
    std::fprintf(stderr,
                 "viz: refusing to bind '%s': the visualizer is unauthenticated"
                 " and binds loopback only\n",
                 opts.host.c_str());
    return 2;
  }

  // Bound the work a single request can cause.
  svr.set_payload_max_length(64 * 1024);
  svr.set_read_timeout(10, 0);
  svr.set_write_timeout(10, 0);

  svr.Get("/", [](const httplib::Request &, httplib::Response &res) {
    const char *html = embedded_ui_data();
    const std::size_t n = embedded_ui_size();
    if (html && n > 0)
      res.set_content(html, n, "text/html; charset=utf-8");
    else
      res.set_content(PLACEHOLDER_HTML, "text/html; charset=utf-8");
  });

  svr.Get("/api/health", [](const httplib::Request &, httplib::Response &res) {
    res.set_content("{\"ok\":true}", "application/json");
  });

  // Long-poll: block until the session publishes something newer than `since`,
  // then return one framed binary message. The client asks for the next frame
  // when it has finished rendering, so it applies natural backpressure and can
  // never fall behind a queue.
  svr.Get("/api/state",
          [&session](const httplib::Request &req, httplib::Response &res) {
            uint64_t since = 0;
            if (req.has_param("since")) {
              try {
                since = std::stoull(req.get_param_value("since"));
              } catch (const std::exception &) {
                since = 0;
              }
            }
            const Snapshot s = session.wait_for(since, 5000);
            res.set_content(encode_state(s), "application/octet-stream");
            res.set_header("Cache-Control", "no-store");
          });

  svr.Get("/api/net",
          [&session](const httplib::Request &, httplib::Response &res) {
            const Snapshot s = session.snapshot();
            res.set_content(encode_net_info(session.net(), s.nnueActive),
                            "application/json");
          });

  svr.Post("/api/control",
           [&session](const httplib::Request &req, httplib::Response &res) {
             int status = 200;
             const std::string body = handle_control(session, req.body, status);
             res.status = status;
             res.set_content(body, "application/json");
           });

  const std::string url =
      "http://" + opts.host + ":" + std::to_string(opts.port) + "/";
  std::printf("Vector Scope listening on %s\n", url.c_str());
  std::fflush(stdout);
  if (opts.openBrowser)
    try_open_browser(url);

  if (!svr.listen(opts.host, opts.port)) {
    std::fprintf(stderr, "viz: could not bind %s:%d (port already in use?)\n",
                 opts.host.c_str(), opts.port);
    return 2;
  }
  return 0;
}

} // namespace Viz
