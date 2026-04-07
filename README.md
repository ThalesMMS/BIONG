# Simulador neuro-modular de aranha com predador explícito e interfaces padronizadas

Implementação do simulador da aranha com três mudanças estruturais centrais:

1. inclusão de um **predador explícito** (`lizard`) no mundo;
2. troca do controle motor abstrato por uma **saída locomotora primitiva** (`cima`, `baixo`, `esquerda`, `direita`, `parada`);
3. definição de **interfaces padrão de entrada/saída** para cada centro/córtex, para reduzir acoplamento excessivo entre as redes.

Na versão atual, esse núcleo foi estendido com:

- **perfis de recompensa** (`classic` e `ecological`);
- **geometria real de abrigo** com entrada, interior e zona profunda;
- **máquinas de estado do lagarto** (`PATROL`, `ORIENT`, `INVESTIGATE`, `CHASE`, `WAIT`, `RECOVER`);
- **memória operacional do lagarto** para alvo investigado, janela de ambush, streak de perseguição e recuperação;
- **memória explícita da aranha** para comida, predador, abrigo seguro e rota de fuga;
- **templates de mapa** e **cenários determinísticos** para avaliação comportamental.

O sistema continua com redes neurais pequenas em NumPy, aprendizado online e módulos independentes, mas agora a arquitetura está mais alinhada com a biologia simplificada do organismo e com a modularização funcional pedida.

## O que mudou

### 1. Predador explícito: lagarto (`lizard`)

O risco noturno probabilístico foi substituído por uma entidade concreta no ambiente.

Características do lagarto:

- ocupa uma célula própria no grid;
- possui **visão limitada**;
- move-se **mais lentamente** do que a aranha;
- **não entra na caverna/abrigo**;
- ao encostar na aranha causa **dor** e perda de saúde;
- a aranha pode fugir caso continue se movendo, porque o lagarto é mais lento.

### 2. Saída motora primitiva

O córtex motor não escolhe mais ações semânticas como `MOVE_TO_FOOD` ou `MOVE_TO_SHELTER`.

Agora o espaço motor é:

- `MOVE_UP`
- `MOVE_DOWN`
- `MOVE_LEFT`
- `MOVE_RIGHT`
- `STAY`

Com isso, o simulador fica menos acoplado a comportamentos “prontos” e mais próximo de um organismo simples controlando locomoção básica.

### 3. Comer e dormir como comportamentos situados/autonômicos

Como a saída motora foi reduzida à locomoção, **alimentação** e **repouso** passam a emergir do contexto espacial:

- ao alcançar comida, a aranha alimenta-se automaticamente;
- ao alcançar o abrigo durante fadiga/noite, a aranha recupera-se automaticamente;
- ficar parada ainda pode potencializar alimentação/repouso, mas não é mais um pré-requisito rígido.

Isso simplifica a biologia do agente e evita que a rede motora precise codificar “comer” e “dormir” como verbos separados.

## Arquitetura modular

O cérebro agora tem cinco módulos propositores + `action_center` + córtex motor:

1. **visual_cortex**
2. **sensory_cortex**
3. **hunger_center**
4. **sleep_center**
5. **alert_center**
6. **action_center**
7. **motor_cortex**

As cinco primeiras redes produzem **propostas locomotoras** sobre o mesmo espaço de saída. O `action_center` arbitra essas propostas em um espaço locomotor explícito, e o `motor_cortex` passa a atuar como executor/corretor locomotor final.

Além disso, foi introduzido um mecanismo simples de redução de coadaptação:

- **interfaces fixas e nomeadas** por módulo;
- **module dropout** durante treinamento;
- **reflexos locais por módulo**, definidos a partir da própria interface do módulo;
- **alvos auxiliares por módulo** (“reflex targets”) para que cada rede aprenda sua função específica em vez de depender excessivamente das demais.

### Responsabilidades por camada

- `spider_cortex_sim/world.py`: mantém a dinâmica ecológica, o estado corporal e a memória explícita observável.
- `spider_cortex_sim/interfaces.py`: define os contratos nomeados de entrada e saída que ligam mundo e cérebro.
- `spider_cortex_sim/modules.py`: executa apenas as redes proponentes e devolve propostas locomotoras por módulo.
- `spider_cortex_sim/agent.py`: aplica reflexos locais interpretáveis, calcula alvos auxiliares auditáveis, arbitra em `action_center` e executa a correção final em `motor_cortex`.

## Interfaces padrão de entrada/saída

A saída comum de todos os módulos propositores é sempre o mesmo vetor locomotor:

```text
[MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY]
```

### 1. `visual_cortex`

Entrada (`visual`):

- comida visível?
- direção relativa da comida visível
- abrigo visível?
- direção relativa do abrigo visível
- predador visível?
- direção relativa do predador visível
- dia
- noite

Função: usar apenas objetos presentes no campo visual limitado.

### 2. `sensory_cortex`

Entrada (`sensory`):

- dor recente causada pelo predador
- contato físico recente com o predador
- saúde
- fome
- fadiga
- intensidade do cheiro de comida
- gradiente do cheiro de comida
- intensidade do cheiro do predador
- gradiente do cheiro do predador
- luminosidade

Função: integrar sinais corporais e químicos imediatos.

### 3. `hunger_center`

Entrada (`hunger`):

- fome
- está sobre comida?
- comida visível?
- direção da comida visível
- intensidade do cheiro de comida
- gradiente do cheiro de comida

Função: puxar a locomoção em direção à alimentação.

### 4. `sleep_center`

Entrada (`sleep`):

- fadiga
- fome
- está no abrigo?
- noite
- vetor interno até o abrigo (`home_dx`, `home_dy`, `home_dist`)
- saúde
- dor recente
- fase atual de sono (`sleep_phase_level`)
- continuidade recente do repouso (`rest_streak_norm`)
- dívida de sono (`sleep_debt`)
- profundidade atual no abrigo (`shelter_role_level`)
- memória do último abrigo seguro

Função: promover retorno ao abrigo e repouso, mas levando em conta o estado corporal.

### 5. `alert_center`

Entrada (`alert`):

- predador visível?
- direção relativa do predador visível
- distância ao predador
- intensidade do cheiro do predador
- vetor interno até o abrigo
- dor recente
- contato recente
- está no abrigo?
- noite
- memória da última posição vista do predador
- memória da rota recente de fuga

Função: fuga do predador e priorização do abrigo quando há ameaça.

### 6. `action_center`

Entrada (`action_context`):

- fome
- fadiga
- saúde
- dor recente
- contato recente
- está sobre comida?
- está no abrigo?
- predador visível?
- confiança visual no predador
- distância ao predador
- dia/noite
- último deslocamento
- dívida de sono
- profundidade atual no abrigo

Saída:

- logits sobre `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`, `STAY`
- estimativa de valor do estado

Input efetivo da rede:

- `SpiderBrain._build_action_input()` concatena as propostas locomotoras dos módulos especializados ao vetor bruto de `action_context`.

Função: arbitrar explicitamente entre fome, ameaça, repouso e exploração antes da execução motora.

### 7. `motor_cortex`

Entrada (`motor_context`):

- está sobre comida?
- está no abrigo?
- predador visível?
- confiança visual no predador
- distância ao predador
- dia/noite
- último deslocamento
- profundidade atual no abrigo

Saída:

- logits corretivos sobre `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`, `STAY`

Input efetivo da rede:

- `SpiderBrain._build_motor_input()` concatena a intenção locomotora escolhida pelo `action_center` em one-hot ao vetor bruto de `motor_context`.

Função: especializar a execução locomotora final sem acumular a arbitragem comportamental.

## Como o aprendizado funciona

O treinamento continua **online**, passo a passo.

Em cada tick:

1. o ambiente gera as interfaces sensoriais padronizadas;
2. cada módulo especializado propõe locomoção usando apenas sua própria interface;
3. cada módulo também pode adicionar um pequeno **viés reflexo inato** compatível com sua função;
4. o `action_center` arbitra as propostas e escolhe uma intenção locomotora explícita;
5. o `motor_cortex` corrige/executa essa intenção no contexto local;
6. o ambiente aplica a transição, incluindo comida, repouso e predador;
7. ocorre atualização TD online do ator-crítico no `action_center`, da correção locomotora no `motor_cortex` e um ajuste auxiliar por módulo baseado em seu reflexo local.

## Ambiente

O mundo em grade 2D agora contém:

- **abrigo/caverna** (`H`)
- **papéis espaciais do abrigo** (`entrance`, `inside`, `deep`)
- **paredes/obstáculos** (`#`)
- **terreno com clutter** (`:`)
- **comida** (`F`)
- **aranha** (`A`)
- **lagarto predador** (`L`)
- ciclo de **dia/noite**
- visão limitada
- cheiro de comida e do predador
- homeostase por fome, fadiga, saúde, dor e contato

No render ASCII, se lagarto e aranha ocuparem a mesma célula, aparece `X`.

## Estrutura do projeto

```text
neuro_modular_sim/
├── README.md
├── spider_cortex_sim/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent.py
│   ├── bus.py
│   ├── cli.py
│   ├── gui.py
│   ├── interfaces.py
│   ├── modules.py
│   ├── nn.py
│   ├── simulation.py
│   └── world.py
└── tests/
    ├── test_training.py
    └── test_world.py
```

## Instalação

### 1. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

> **Nota:** `numpy` é obrigatório. `pygame-ce` é opcional, necessário apenas para a interface gráfica (`--gui`).

## Como executar

No diretório raiz do projeto, com o ambiente virtual ativado:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim --episodes 120 --eval-episodes 3 --max-steps 90
```

Com resumo e trace salvos em arquivo:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 120 \
  --eval-episodes 1 \
  --max-steps 90 \
  --summary spider_summary.json \
  --trace spider_trace.jsonl
```

Para renderizar a última avaliação em ASCII:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 120 \
  --eval-episodes 1 \
  --max-steps 90 \
  --render-eval
```

### Perfis de recompensa e mapas

Treinar com o perfil ecológico e um mapa alternativo:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 120 \
  --eval-episodes 3 \
  --max-steps 90 \
  --reward-profile ecological \
  --map-template side_burrow
```

Os templates atualmente disponíveis são:

- `central_burrow`
- `side_burrow`
- `corridor_escape`
- `two_shelters`
- `exposed_feeding_ground`
- `entrance_funnel`

`corridor_escape` usa agora terreno `NARROW`, que representa passagem estreita:

- reduz a confiança visual;
- aumenta risco quando o predador está próximo;
- adiciona custo/fadiga próprios.

Os layouts adicionais foram feitos para produzir episódios ecologicamente diferentes:

- `two_shelters`: dois abrigos profundos competindo com uma zona central de transição;
- `exposed_feeding_ground`: comida concentrada em área aberta, com aproximação mais exposta;
- `entrance_funnel`: entrada do abrigo comprimida por um gargalo que favorece bloqueio e ambush.

Os perfis de recompensa são:

- `classic`: mais guiado, melhor para treino rápido e baseline;
- `ecological`: menos shaping direto e mais custo vindo da dinâmica do mundo.

### Cenários determinísticos

Executar um cenário específico:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --scenario night_rest
```

Executar a suíte completa de cenários:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --scenario-suite
```

Cenários disponíveis:

- `night_rest`
- `predator_edge`
- `entrance_ambush`
- `open_field_foraging`
- `shelter_blockade`
- `recover_after_failed_chase`
- `corridor_gauntlet`
- `two_shelter_tradeoff`
- `exposed_day_foraging`
- `food_deprivation`

A suíte agora combina cenários e mapas específicos, por exemplo:

- `entrance_ambush`, `shelter_blockade` e `recover_after_failed_chase` usam `entrance_funnel`;
- `open_field_foraging` e `exposed_day_foraging` usam `exposed_feeding_ground`;
- `corridor_gauntlet` usa `corridor_escape`;
- `two_shelter_tradeoff` usa `two_shelters`.

### Avaliação comportamental

Executar a suíte comportamental completa com scorecards explícitos:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --behavior-suite \
  --full-summary
```

Executar apenas um cenário comportamental e exportar CSV flat:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --behavior-scenario night_rest \
  --behavior-csv spider_behavior.csv \
  --full-summary
```

O bloco `behavior_evaluation` do `summary` inclui:

- `suite`: scorecards agregados por cenário com `success_rate`, `checks`, `behavior_metrics`, `diagnostics` e `failures`;
- `summary`: taxa de sucesso da suíte e regressões detectadas;
- `comparisons`: matriz opcional por perfil/mapa/seed quando os flags de comparação comportamental são usados.
- `learning_evidence`: comparação opcional entre checkpoint treinado e controles como `random_init`, `reflex_only`, `freeze_half_budget` e `trained_long_budget`.

Para os cenários fracos do benchmark curto, os scorecards agora carregam metadata pública do próprio cenário:

- `diagnostic_focus`: qual ambiguidade o scorecard quer separar;
- `success_interpretation`: como ler um cenário realmente resolvido;
- `failure_interpretation`: como ler um `0.0` sem colapsar tudo em “não aprendeu”;
- `budget_note`: nota curta sobre o que o orçamento `dev` costuma revelar naquele cenário.

O bloco `suite[scenario]["diagnostics"]` resume especialmente:

- `primary_outcome`: desfecho predominante do cenário;
- `outcome_distribution`: distribuição normalizada dos desfechos por episódio;
- `partial_progress_rate`: taxa de episódios com progresso parcial real;
- `died_without_contact_rate`: taxa de mortes sem contato direto com o predador.

Os quatro cenários mais fracos no orçamento curto agora distinguem melhor “falha total” de “sinal parcial útil”:

- `open_field_foraging` e `exposed_day_foraging`: usam `progress_band` para separar avanço, estagnação e regressão espacial.
- `corridor_gauntlet`: destaca o caso “evitou contato, mas ficou estagnado” sem tratá-lo como a mesma falha de uma morte após progresso.
- `food_deprivation`: pode falhar com `success_rate=0.0` mesmo quando há aproximação real da comida; esse caso aparece como `partial_progress_died`.

### Interface gráfica (Pygame)

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --gui \
  --episodes 120 \
  --eval-episodes 3 \
  --max-steps 90 \
  --reward-profile classic \
  --map-template central_burrow
```

A interface agora exibe também:

- o **lagarto predador** no grid;
- papéis do abrigo e terreno do mapa;
- contadores de **contatos**, **avistamentos** e **fugas**;
- posição, modo e alvo atual do lagarto;
- memória explícita da aranha e dívida de sono;
- componentes recentes de recompensa;
- distribuição noturna por papel do abrigo e ocupação dos estados do lagarto.

Atalhos úteis na GUI:

- `V`: overlay de células visíveis/ocluídas;
- `M`: heatmap de cheiro de comida/predador.

## Salvar e carregar o cérebro

Treinar e salvar:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 120 --eval-episodes 3 --max-steps 90 \
  --save-brain spider_brain
```

Carregar e continuar treinando:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 60 --eval-episodes 3 --max-steps 90 \
  --load-brain spider_brain \
  --save-brain spider_brain
```

Carregar apenas módulos específicos:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 60 --eval-episodes 3 --max-steps 90 \
  --load-brain spider_brain \
  --load-modules visual_cortex hunger_center
```

Observação: a arquitetura atual possui uma assinatura de interfaces. Saves antigos, anteriores à padronização atual, são rejeitados com erro explícito de incompatibilidade.
Saves treinados antes da introdução de `sleep_phase` e `rest_streak` também são incompatíveis com a arquitetura atual.
Saves anteriores à arquitetura com `sleep_debt`, papéis do abrigo, sinais de certeza/oclusão visual e memória explícita também são incompatíveis com a versão atual.
O registry versionado das interfaces e a documentação gerada dos contratos atuais estão em [`docs/interfaces.md`](docs/interfaces.md).

## Testes

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

Os testes cobrem:

- shapes das interfaces padronizadas;
- alimentação e repouso contextuais;
- decomposição auditável de recompensa por componente;
- progressão de sono `SETTLING -> RESTING -> DEEP_SLEEP`, dívida de sono e interrupções;
- contato do lagarto com dano real;
- restrição geométrica de abrigo e oclusão;
- memória explícita da aranha;
- templates de mapa e reachability;
- regressões determinísticas de cenário;
- memória guiando fuga e forrageio mesmo após perda de visada;
- runner de cenários e métrica determinística de latência de resposta ao predador;
- atualização online de parâmetros;
- checks leves de treinabilidade em `classic`, `ecological`, comparações e mapa alternativo.

## Métricas e rastreio

O `summary` e o `trace` agora incluem:

- `reward_components` por passo, permitindo auditar a recompensa total sem ler `world.py`;
- ocupação noturna do abrigo e taxa de imobilidade noturna;
- distribuição noturna por papel do abrigo (`outside`, `entrance`, `inside`, `deep`);
- latência média de resposta a encontros com o predador;
- `reward_profile` e `map_template`;
- `config.operational_profile`, com nome, versão e payload efetivo dos thresholds/pesos operacionais ativos;
- `config.budget`, com perfil de orçamento, força do benchmark, seeds resolvidos e overrides explícitos;
- `checkpointing`, quando `--checkpoint-selection best` é usado em workflows comportamentais;
- campos explícitos de certeza/oclusão visual por canal;
- vetores de memória (`dx`, `dy`, `age/ttl`) no metadata do trace;
- ocupação do predador por estado (`PATROL`, `ORIENT`, `INVESTIGATE`, `CHASE`, `WAIT`, `RECOVER`);
- contagem de transições de estado do predador e estado dominante por episódio/cenário;
- deltas de distância para comida e abrigo, para diferenciar episódios de aproximação, fuga e retorno;
- resultados por cenário quando `--scenario` ou `--scenario-suite` são usados.
- scorecards comportamentais por cenário em `behavior_evaluation`, com checks threshold-based reproduzíveis por seed.
- bandas diagnósticas por episódio (`progress_band`, `outcome_band`) nos cenários fracos, para diferenciar regressão, estagnação, progresso parcial e sucesso completo.

Se `--debug-trace` for combinado com `--trace`, cada tick inclui também:

- observação serializada antes/depois da transição;
- componentes de recompensa;
- vetores de memória já normalizados;
- logits neurais por módulo antes do reflexo, delta reflexo aplicado e logits pós-reflexo;
- `effective_reflex_scale`, `module_reflex_override`, `module_reflex_dominance` e `final_reflex_override`;
- estado interno completo do predador.

A saída curta da CLI também imprime esses agregados de avaliação para facilitar comparação rápida de baselines.

Para imprimir o `summary` integral no stdout, use `--full-summary`.

O perfil operacional ativo pode ser selecionado pela CLI com `--operational-profile`. O perfil inicial `default_v1` preserva exatamente os thresholds e pesos atuais de reflexos, percepção e heurísticas de reward; esse mesmo bloco também é persistido no `metadata.json` de `save/load` e nas linhas exportadas em `behavior_csv` via `operational_profile` e `operational_profile_version`.

O envelope experimental de ruído agora é explícito e independente do perfil operacional. Use `--noise-profile {none,low,medium,high}` para ativar ruído controlado em percepção visual, olfato, execução motora, spawn/respawn e desempates do predador. O valor default `none` preserva o caminho reproduzível anterior; o `summary` passa a registrar `config.noise_profile` com o payload resolvido de cada canal.

Os perfis de orçamento também são explícitos na CLI e nos artefatos:

- `smoke`: sanity check rápido/CI (`6` episódios, `1` eval, `60` passos, seed `7`);
- `dev`: benchmark local curto e reproduzível (`12` episódios, `2` eval, `90` passos, seeds `7/17/29`);
- `report`: workflow canônico para conclusões mais fortes (`24` episódios, `4` eval, `120` passos, `2` repetições por cenário, seeds `7/17/29/41/53`).

Sem `--budget-profile`, a execução continua possível em modo `custom`; nesse caso, `summary["config"]["budget"]` registra os valores efetivos e qualquer override manual aplicado por `--episodes`, `--eval-episodes`, `--max-steps`, `--scenario-episodes`, `--behavior-seeds`, `--ablation-seeds` ou `--checkpoint-interval`.

O `behavior_csv` também passa a carregar `budget_profile`, `benchmark_strength`, `checkpoint_source` e `noise_profile`, além de `operational_profile` e da metadata de ablação já existente.
Ele também exporta `architecture_version` e `architecture_fingerprint` para rastreabilidade entre runs e checkpoints.
Ele agora inclui também `scenario_description`, `scenario_objective`, `scenario_focus`, além de `metric_progress_band`, `metric_outcome_band` e dos booleanos diagnósticos dos cenários fracos.

## Perfis de orçamento

Comandos canônicos por perfil:

```bash
# smoke: sanity/CI
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --behavior-suite --full-summary

# dev: benchmark local rápido
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite --full-summary

# report: benchmark forte + seleção automática do melhor checkpoint
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile report \
  --checkpoint-selection best \
  --ablation-suite --full-summary
```

Custos esperados:

- `smoke`: segundos, pensado para sanity check;
- `dev`: dezenas de segundos locais, pensado para iteração;
- `report`: mais caro, pensado para documentação/relato e não para loop rápido.

### Workflow de comparação

Comparar perfis de recompensa no mapa atual:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --compare-profiles --full-summary
```

Comparar mapas no perfil atual:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --reward-profile ecological \
  --compare-maps --full-summary
```

Gerar summary + debug trace enriquecido:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --compare-profiles --compare-maps \
  --summary spider_summary_compare.json \
  --trace spider_trace_debug.jsonl \
  --debug-trace
```

### Workflow de comparação comportamental

Comparar a suíte comportamental entre perfis:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --behavior-compare-profiles --full-summary
```

Comparar a suíte comportamental entre mapas e exportar CSV:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --reward-profile ecological \
  --behavior-compare-maps \
  --behavior-csv spider_behavior_compare.csv \
  --full-summary
```

### Análise offline dos artefatos

O projeto agora inclui um runner separado para transformar `summary.json`, `trace.jsonl` e `behavior_csv` em um bundle de análise offline sem dependências novas.

Comando canônico:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim.offline_analysis \
  --summary spider_summary_compare.json \
  --trace spider_trace_debug.jsonl \
  --behavior-csv spider_behavior_compare.csv \
  --output-dir offline_analysis
```

Regras do runner:

- `--output-dir` é obrigatório;
- ao menos um entre `--summary`, `--trace` e `--behavior-csv` é obrigatório;
- o relatório sempre é emitido, mesmo com entrada parcial;
- quando alguma leitura depende de `--debug-trace`, isso aparece como limitação em `report.md` e `report.json`, em vez de abortar a execução.

Artefatos gerados:

- `report.md`
- `report.json`
- `training_eval.svg`
- `scenario_success.svg`
- `scenario_checks.csv`
- `reward_components.csv`
- `ablation_comparison.svg`, quando houver payload de ablação ou CSV suficiente para reconstruí-lo
- `reflex_frequency.svg`, quando houver `trace`

O relatório consolida:

- curva de treino/avaliação quando o `summary` traz histórico suficiente;
- sucesso por cenário e checks com falha a partir de `behavior_evaluation.suite` ou, na falta dele, do `behavior_csv`;
- comparações entre perfis, mapas e variantes de ablação;
- métricas arquiteturais já exportadas, como `mean_reward_components`, `mean_night_role_distribution`, `mean_predator_state_occupancy`, `mean_predator_mode_transitions`, `mean_food_distance_delta` e `mean_shelter_distance_delta`;
- frequência de reflexos por módulo a partir de `trace.messages[*].payload.reflex`, enriquecida por `debug.reflexes` quando o trace foi gerado com `--debug-trace`.

### Workflow de ablações

Comparar a referência modular com a suíte canônica de ablações:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --behavior-csv spider_ablation_rows.csv \
  --full-summary
```

Executar treino com annealing linear de reflexo e registrar a comparação pós-treino com `eval_reflex_scale=0.0`:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --episodes 12 \
  --eval-episodes 2 \
  --reflex-scale 1.0 \
  --reflex-anneal-final-scale 0.25 \
  --summary spider_reflex_schedule_summary.json \
  --full-summary
```

Comparar apenas a baseline monolítica contra a referência modular:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-variant monolithic_policy \
  --behavior-scenario night_rest \
  --full-summary
```

O bloco `behavior_evaluation.ablations` do `summary` inclui:

- `reference_variant`: variante modular usada como referência;
- `variants`: payload compacto por variante, incluindo configuração, suíte, métricas legadas e `without_reflex_support`;
- `deltas_vs_reference`: diferença de `scenario_success_rate`, `episode_success_rate` e `success_rate` por cenário em relação a `modular_full`.

As variantes canônicas agora incluem também o sweep gradual `reflex_scale_0_25`, `reflex_scale_0_50` e `reflex_scale_0_75`.

As linhas exportadas em `behavior_csv` agora carregam também:

- `reflex_scale`;
- `reflex_anneal_final_scale`;
- `eval_reflex_scale`.

## Learning Evidence

Rodar a suíte de `learning_evidence` no orçamento smoke:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --learning-evidence \
  --behavior-scenario night_rest \
  --behavior-csv spider_learning_evidence_rows.csv \
  --full-summary
```

Rodar a comparação canônica curto vs longo:

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

O bloco `behavior_evaluation.learning_evidence` inclui:

- `reference_condition`: hoje fixado em `trained_final`;
- `conditions`: payload compacto por condição, com `policy_mode`, `train_episodes`, `checkpoint_source` e a suíte comportamental agregada;
- `deltas_vs_reference`: deltas de `scenario_success_rate`, `episode_success_rate` e `mean_reward` contra `trained_final`;
- `evidence_summary`: gate objetivo baseado em `scenario_success_rate`, comparando `trained_final` contra `random_init` e `reflex_only`, com `trained_without_reflex_support` registrado apenas como sinal complementar.

As linhas exportadas em `behavior_csv` passam a carregar também:

- `learning_evidence_condition`;
- `learning_evidence_policy_mode`;
- `learning_evidence_train_episodes`;
- `learning_evidence_frozen_after_episode`;
- `learning_evidence_checkpoint_source`;
- `learning_evidence_budget_profile`;
- `learning_evidence_budget_benchmark_strength`.

O workflow completo, com definições das variantes e uma tabela de resultados check-in, está em [docs/ablation_workflow.md](docs/ablation_workflow.md).

## Comandos de benchmark

Baseline clássico:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile report \
  --reward-profile classic --map-template central_burrow \
  --summary spider_summary_classic.json
```

Comparação ecológica:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile report \
  --reward-profile ecological --map-template central_burrow \
  --summary spider_summary_ecological.json
```

Mapa alternativo com cenários:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --map-template side_burrow \
  --scenario-suite \
  --summary spider_summary_side_burrow.json
```

## Observações de modelagem

- O sistema é **biologicamente inspirado**, não biologicamente fiel.
- O “vetor de retorno ao abrigo” é uma simplificação de propriocepção / memória espacial mínima.
- A memória explícita é **world-owned**: idade, alvo e TTL são mantidos pelo ambiente e apenas consumidos pelos módulos corticais via observação.
- Os reflexos locais por módulo funcionam como uma forma de “comportamento inato”, sobre a qual o aprendizado online faz refinamento, e agora aparecem no trace/debug com ação-alvo, motivo e sinais disparadores.
- Não há um centro coringa gigante integrando tudo; cada módulo recebe apenas sua interface própria e emite apenas proposta locomotora padronizada.

## Extensões naturais

1. adicionar memória recorrente por módulo;
2. introduzir múltiplos predadores com nichos sensoriais diferentes;
3. modelar campo visual orientado em vez de visão omnidirecional curta;
4. separar locomoção em marcha/velocidade/orientação corporal;
5. migrar as redes para PyTorch preservando a mesma assinatura modular.
