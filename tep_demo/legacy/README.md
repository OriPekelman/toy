# tep_demo/legacy/

Superseded Tep+Spinel serving demos. The canonical serving path is the `toy`
CLI (`toy serve <model.gguf>`); `tep_demo/` keeps only the minimal Tep demo
(`hello_api`) + the GPT-2 OpenAI server (`openai_api`, pending toy#30).

- `inference_api.rb` — a second serving demo (toy random-init
  `FullForwardFFICache`, `/generate?n=N`). Still builds via `make tep_demo/api`.
