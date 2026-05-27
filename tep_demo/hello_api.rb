# Minimal tep+spinel sanity test -- no model, just verifies that the
# build pipeline works for our project layout.

require_relative "../vendor/spinel/deps"

class HelloHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "text/plain"
    "hello from tep + spinel\n"
  end
end

Tep.get "/", HelloHandler.new
Tep.run!(4567, 1, false)
