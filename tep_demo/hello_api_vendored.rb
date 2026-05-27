# Twin of tep_demo/hello_api.rb that requires Tep via the
# spinelgems-vendored path instead of the rsync'd _tep_lib/. Used as
# the parity check during the spinelgems adoption: build both, diff
# binary behaviour, then retire prep/sync_tep.rb when parity holds.
#
# See docs/roadmap/spinelgems-tep-adoption-2026-05-27.md for the plan.

require_relative "../vendor/spinel/deps"

class HelloHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "text/plain"
    "hello from tep + spinel (vendored path)\n"
  end
end

Tep.get "/", HelloHandler.new
Tep.run!(4567, 1, false)
