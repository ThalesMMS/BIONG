# Workflow de ablações

Este repositório agora expõe uma suíte de ablações reproduzível para comparar a arquitetura modular atual com variantes estruturais e com uma baseline monolítica comparável.

## Variantes canônicas

- `modular_full`: arquitetura modular completa, com `module_dropout`, reflexos locais e alvos auxiliares.
- `no_module_dropout`: mantém a arquitetura modular, mas zera o dropout entre módulos.
- `no_module_reflexes`: mantém os módulos, mas desliga tanto a injeção de logits reflexos quanto os alvos auxiliares derivados desses reflexos.
- `reflex_scale_0_25`, `reflex_scale_0_50`, `reflex_scale_0_75`: mantêm a arquitetura modular, mas reduzem gradualmente a intensidade reflexa global.
- `drop_visual_cortex`, `drop_sensory_cortex`, `drop_hunger_center`, `drop_sleep_center`, `drop_alert_center`: removem um módulo por vez da proposta locomotora e do aprendizado.
- `monolithic_policy`: concatena as observações modulares em um único vetor e usa uma única rede propositora antes do mesmo `action_center` e do mesmo `motor_cortex` da arquitetura modular.

## Topologia após `action_center`

- As variantes modulares continuam diferindo na etapa propositora e no suporte reflexo.
- A arbitragem explícita agora acontece em `action_center`.
- O `motor_cortex` passa a ser um estágio de correção/execução locomotora, não o árbitro principal.
- A nota curta de design da topologia nova está em `docs/action_center_design.md`.

## Comandos principais

Rodar a suíte canônica completa e exportar `summary` + CSV:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --summary spider_ablation_summary.json \
  --behavior-csv spider_ablation_rows.csv \
  --full-summary
```

Rodar apenas uma variante contra a referência modular em um cenário específico:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-variant monolithic_policy \
  --behavior-scenario night_rest \
  --full-summary
```

## Learning evidence

O repositório também expõe um workflow separado para medir se o checkpoint treinado supera controles apropriados sem depender só de narrativa qualitativa.

Comando smoke:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --learning-evidence \
  --behavior-scenario night_rest \
  --full-summary
```

Comando canônico curto vs longo:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --learning-evidence \
  --learning-evidence-long-budget-profile report \
  --behavior-suite \
  --summary spider_learning_evidence_summary.json \
  --behavior-csv spider_learning_evidence_rows.csv \
  --full-summary
```

Condições canônicas:

- `trained_final`
- `trained_without_reflex_support`
- `random_init`
- `reflex_only`
- `freeze_half_budget`
- `trained_long_budget`

Leitura prática:

- `summary["behavior_evaluation"]["learning_evidence"]["reference_condition"]` é `trained_final`.
- `evidence_summary["has_learning_evidence"]` usa apenas `scenario_success_rate` como gate principal.
- `trained_without_reflex_support` não entra no gate; ele serve para medir dependência residual de reflexos.
- O CSV exportado inclui `learning_evidence_condition`, `learning_evidence_policy_mode`, `learning_evidence_train_episodes`, `learning_evidence_frozen_after_episode`, `learning_evidence_checkpoint_source`, `learning_evidence_budget_profile` e `learning_evidence_budget_benchmark_strength`.

## Pós-processamento offline

Depois de gerar `summary`, `trace` e `behavior_csv`, o runner offline consegue montar um pacote comparável sem depender de notebooks ou leitura manual de JSON.

Exemplo completo:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --summary spider_ablation_summary.json \
  --trace spider_ablation_trace.jsonl \
  --debug-trace \
  --behavior-csv spider_ablation_rows.csv \
  --full-summary

PYTHONPATH=. python3 -m spider_cortex_sim.offline_analysis \
  --summary spider_ablation_summary.json \
  --trace spider_ablation_trace.jsonl \
  --behavior-csv spider_ablation_rows.csv \
  --output-dir spider_ablation_report
```

Saídas esperadas em `spider_ablation_report/`:

- `report.md`
- `report.json`
- `training_eval.svg`
- `scenario_success.svg`
- `scenario_checks.csv`
- `reward_components.csv`
- `ablation_comparison.svg`
- `reflex_frequency.svg`

Leitura prática:

- `ablation_comparison.svg` resume `scenario_success_rate` por variante quando `behavior_evaluation.ablations` existe ou pode ser reconstruído do CSV;
- `scenario_success.svg` e `scenario_checks.csv` ajudam a localizar regressões por cenário/check sem abrir o JSON inteiro;
- `reward_components.csv` cruza componentes de reward agregados do `summary` com totais observados no `trace`;
- `reflex_frequency.svg` usa `trace.messages[*].payload.reflex` como base e fica mais rico quando o trace foi produzido com `--debug-trace`.

Se algum bloco não existir, o runner não falha: ele registra a ausência em `report.md` e `report.json`.

Observações do workflow:

- Se nenhum `--behavior-scenario` for informado junto com `--ablation-suite` ou `--ablation-variant`, a CLI usa automaticamente a suíte comportamental completa.
- `summary["behavior_evaluation"]["ablations"]` contém a referência usada, as variantes avaliadas e os deltas contra `modular_full`.
- Cada variante também publica `without_reflex_support`, que reavalia o mesmo checkpoint com `eval_reflex_scale=0.0`.
- O CSV exportado inclui `ablation_variant`, `ablation_architecture`, `reflex_scale`, `reflex_anneal_final_scale`, `eval_reflex_scale`, `budget_profile`, `benchmark_strength`, `checkpoint_source`, `operational_profile`, `operational_profile_version` e `noise_profile` em todas as linhas geradas pela rotina de ablação.
- O `summary["config"]["operational_profile"]` registra o nome, a versão e o payload efetivo do perfil operacional ativo; a CLI permite fixá-lo com `--operational-profile default_v1`.
- O `summary["config"]["noise_profile"]` registra o nome e o payload efetivo do perfil de ruído ativo; a CLI permite fixá-lo com `--noise-profile none|low|medium|high`.
- O `summary["config"]["budget"]` registra o perfil resolvido (`smoke`, `dev`, `report` ou `custom`), a força do benchmark, os seeds usados por comparação e qualquer override manual.
- `--checkpoint-selection best` avalia checkpoints intermediários pela suíte comportamental, escolhe o melhor por `scenario_success_rate` e persiste apenas `best/` e `last/` quando `--checkpoint-dir` é fornecido.

## Sweep gradual de reflexo

O workflow de ablação agora permite medir dominância reflexa de forma gradual, e não apenas binária.

Os principais sinais agregados ficam em `summary["evaluation"]` e nos blocos legados por cenário:

- `mean_reflex_usage_rate`
- `mean_final_reflex_override_rate`
- `mean_reflex_dominance`
- `mean_module_reflex_usage_rate`
- `mean_module_reflex_override_rate`
- `mean_module_reflex_dominance`

Para treinos fora da suíte canônica, a CLI também aceita:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --episodes 12 \
  --eval-episodes 2 \
  --reflex-scale 1.0 \
  --module-reflex-scale alert_center=0.5 \
  --reflex-anneal-final-scale 0.25 \
  --summary spider_reflex_schedule_summary.json \
  --full-summary
```

Nesse caso, `summary["config"]["reflex_schedule"]` registra o schedule linear resolvido e `summary["evaluation_without_reflex_support"]` registra a mesma avaliação com reflexos desligados.

## Perfis recomendados

- `smoke`: sanity/CI, com `6` episódios, `1` eval, `60` passos e seed única `7`.
- `dev`: benchmark local rápido, com `12` episódios, `2` eval, `90` passos, `scenario_episodes=1` e seeds `7/17/29`.
- `report`: benchmark forte, com `24` episódios, `4` eval, `120` passos, `scenario_episodes=2` e seeds `7/17/29/41/53`.

Comandos canônicos:

```bash
# sanity / CI
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --ablation-variant monolithic_policy \
  --behavior-scenario night_rest \
  --full-summary

# benchmark local rápido
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --full-summary

# benchmark forte com checkpoint selection
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile report \
  --checkpoint-selection best \
  --ablation-suite \
  --full-summary
```

## Resultado canônico check-in

Comando usado para gerar a tabela abaixo:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --full-summary
```

Parâmetros implícitos:

- `budget_profile=dev`
- `reward_profile=classic`
- `map_template=central_burrow`
- `ablation_seeds=(7, 17, 29)`
- `scenario_episodes=1`
- suíte completa de cenários comportamentais

### Resumo por variante

| variante | arquitetura | suite success | episode success | delta suite vs ref | delta episode vs ref |
| --- | --- | ---: | ---: | ---: | ---: |
| `modular_full` | `modular` | 0.30 | 0.37 | +0.00 | +0.00 |
| `no_module_dropout` | `modular` | 0.30 | 0.47 | +0.00 | +0.10 |
| `no_module_reflexes` | `modular` | 0.10 | 0.33 | -0.20 | -0.03 |
| `reflex_scale_0_25` | `modular` | 0.10 | 0.30 | -0.20 | -0.07 |
| `reflex_scale_0_50` | `modular` | 0.30 | 0.37 | +0.00 | +0.00 |
| `reflex_scale_0_75` | `modular` | 0.10 | 0.30 | -0.20 | -0.07 |
| `drop_visual_cortex` | `modular` | 0.40 | 0.47 | +0.10 | +0.10 |
| `drop_sensory_cortex` | `modular` | 0.30 | 0.50 | +0.00 | +0.13 |
| `drop_hunger_center` | `modular` | 0.40 | 0.50 | +0.10 | +0.13 |
| `drop_sleep_center` | `modular` | 0.10 | 0.30 | -0.20 | -0.07 |
| `drop_alert_center` | `modular` | 0.10 | 0.37 | -0.20 | +0.00 |
| `monolithic_policy` | `monolithic` | 0.10 | 0.30 | -0.20 | -0.07 |

### Comparação por cenário

Legenda curta: `full=modular_full`, `no_drop=no_module_dropout`, `no_reflex=no_module_reflexes`, `r025=reflex_scale_0_25`, `r050=reflex_scale_0_50`, `r075=reflex_scale_0_75`, `d_visual=drop_visual_cortex`, `d_sensory=drop_sensory_cortex`, `d_hunger=drop_hunger_center`, `d_sleep=drop_sleep_center`, `d_alert=drop_alert_center`, `mono=monolithic_policy`.

| cenário | full | no_drop | no_reflex | r025 | r050 | r075 | d_visual | d_sensory | d_hunger | d_sleep | d_alert | mono |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `night_rest` | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 | 0.00 | 0.67 | 0.00 |
| `predator_edge` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `entrance_ambush` | 0.33 | 1.00 | 0.67 | 0.67 | 1.00 | 0.33 | 0.67 | 0.67 | 1.00 | 0.33 | 0.33 | 0.67 |
| `open_field_foraging` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `shelter_blockade` | 0.33 | 0.67 | 0.67 | 0.33 | 0.67 | 0.33 | 1.00 | 1.00 | 1.00 | 0.33 | 0.33 | 0.33 |
| `recover_after_failed_chase` | 1.00 | 0.67 | 0.67 | 0.67 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | 0.67 |
| `corridor_gauntlet` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `two_shelter_tradeoff` | 1.00 | 1.00 | 0.33 | 0.33 | 0.00 | 0.67 | 1.00 | 0.67 | 0.33 | 0.67 | 0.67 | 0.33 |
| `exposed_day_foraging` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `food_deprivation` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Leitura rápida

- O experimento já produz evidência estrutural: desligar reflexos auxiliares (`no_module_reflexes`), remover `sleep_center`, remover `alert_center` ou trocar por uma baseline `monolithic_policy` piora a taxa de sucesso da suíte em `0.20` neste orçamento curto.
- O sweep `reflex_scale_*` permite medir se a política degrada de forma suave, colapsa cedo ou se continua dependente de override reflexo mesmo após treino.
- Em traces/debug, a leitura correta agora é: propostas modulares ou monolítica -> `action_center` -> correção do `motor_cortex` -> (opcional) `final_reflex_override` -> ação final.
- O ganho não é uniformemente positivo para toda forma de modularidade: com `12` episódios de treino, algumas ablações por remoção (`drop_visual_cortex`, `drop_hunger_center`) sobem no score agregado. Isso é um resultado útil, não um problema de documentação.
- A tabela por cenário mostra onde o benchmark ainda é fraco: `open_field_foraging`, `corridor_gauntlet`, `exposed_day_foraging` e `food_deprivation` continuam em `0.00` para todas as variantes nesta configuração curta. Esses cenários continuam bons candidatos para aumentar orçamento de treino ou revisar checks.

## Leitura diagnóstica dos `0.00`

Os scorecards desses quatro cenários agora incluem `diagnostics`, `progress_band` e `outcome_band` para evitar que todo `0.00` seja lido como a mesma falha.

- `open_field_foraging`: `progress_band=regressed` indica deslocamento para longe da comida; `outcome_band=regressed_and_died` separa isso de mera estagnação.
- `corridor_gauntlet`: quando `corridor_avoids_contact` passa mas o cenário continua em `0.00`, o `outcome_band` distingue estagnação segura de morte após progresso parcial.
- `exposed_day_foraging`: `progress_band=regressed` costuma significar recuo ou trajetória defensiva que não virou progresso alimentar.
- `food_deprivation`: `approaches_food` pode passar e o cenário ainda ficar em `0.00`; esse caso agora aparece como `partial_progress_died`, sinalizando aproximação real sem recuperação homeostática suficiente.

Na prática:

- `suite[scenario]["diagnostics"]["primary_outcome"]` resume o desfecho dominante.
- `suite[scenario]["diagnostics"]["outcome_distribution"]` mostra a mistura de desfechos por episódio.
- `partial_progress_rate` ajuda a separar cenário mal calibrado de cenário realmente sem sinal útil.
- No CSV, use `scenario_focus`, `metric_progress_band` e `metric_outcome_band` para filtrar esses casos sem abrir o JSON completo.
