# L4 — Recipes

A Recipe is a training plan — one or more stages, each composing an
Arch + a Trainer + a DataSpec (+ optional Decoder + Eval).

## Roster (target post-P2)

| File | Class | Stages | Notes |
| --- | --- | --- | --- |
| `from_scratch.rb` | `Toy::LLM::Recipes::FromScratch` | 1 | Random init → AdamW on CE. Default recipe for new archs. |
| `lora.rb` | `Toy::LLM::Recipes::LoRA` | 1 | Frozen base + LoRA adapter on Q. Default recipe for fine-tuning pretrained checkpoints. |
| `warm_start.rb` | `Toy::LLM::Recipes::WarmStart` | 2 | Stage 1: load pretrained. Stage 2: full fine-tune. |
| `curriculum.rb` | `Toy::LLM::Recipes::Curriculum` | N | Template recipe showing the multi-stage shape. Each stage can swap dataset / context length / trainer. |

## Contract

```ruby
class Toy::LLM::Recipes::FromScratch
  # Stages are evaluated lazily; each returns a stage-cfg the
  # driver can dispatch on. A single-stage recipe yields once.
  def each_stage(cfg)
    yield Stage.new(
      arch:    arch_for(cfg),
      trainer: Toy::Trainers::AdamW.new(cfg.optim),
      data:    Toy::DataSpecs.from(cfg.data),
      eval:    Toy::Evals::CrossEntropy.new(cfg.eval),
      stop:    cfg.steps,
    )
  end
end
```

## CurriculumRecipe shape (template)

```ruby
class Toy::LLM::Recipes::Curriculum
  def each_stage(cfg)
    # Stage 1 — short context, large LR.
    yield Stage.new(arch: arch_for(cfg).with_hyper(:t_seq, 512),
                    trainer: trainer_for(cfg, lr: 3e-3),
                    data:    short_data(cfg),
                    stop:    5_000)

    # Stage 2 — long context, decayed LR.
    yield Stage.new(arch: arch_for(cfg).with_hyper(:t_seq, 2048),
                    trainer: trainer_for(cfg, lr: 5e-4),
                    data:    long_data(cfg),
                    stop:    20_000)
  end
end
```

## What lives on the RECIPE

- Stage sequence (1..N stages).
- Optimizer choice + schedule.
- Data progression across stages.
- Eval choices.
- Stop conditions.

## What does NOT live here

- Card composition operators (those live on `Toy::Card`).
- Backend dispatch (that's the session / Kind).

This file is a contract sketch. Real entries land in P2.6.
