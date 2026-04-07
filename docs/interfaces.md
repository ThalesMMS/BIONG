# Interfaces Padronizadas

<!-- Generated from spider_cortex_sim.interfaces.render_interfaces_markdown(); do not edit manually. -->

- Schema version: `1`
- Registry fingerprint: `b9580719ccb8e4d655e59a3a488aa5fbecb535b08ccb675395a7c8c806ef9fb9`
- Proposal interfaces: `visual_cortex, sensory_cortex, hunger_center, sleep_center, alert_center`
- Context interfaces: `action_center_context, motor_cortex_context`

## Compatibilidade

- A política atual é `exact_match_required` para save/load.
- Mudanças em nome, `observation_key`, ordem dos sinais, outputs ou versão exigem incompatibilidade explícita.
- Não existe migração automática de checkpoints antigos.

## `visual_cortex`

- Observation key: `visual`
- Role: `proposal`
- Version: `1`
- Description: Propositor visual orientado por comida, abrigo e predador dentro do campo visual local.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `food_visible` | -1.0 | 1.0 | 1 se há comida detectada com confiança suficiente no campo visual. |
| 2 | `food_certainty` | 0.0 | 1.0 | Confiança visual na comida percebida. |
| 3 | `food_occluded` | 0.0 | 1.0 | 1 se a comida mais próxima está em alcance, mas ocluída. |
| 4 | `food_dx` | -1.0 | 1.0 | Deslocamento horizontal relativo da comida detectada. |
| 5 | `food_dy` | -1.0 | 1.0 | Deslocamento vertical relativo da comida detectada. |
| 6 | `shelter_visible` | -1.0 | 1.0 | 1 se o abrigo está visível com confiança suficiente. |
| 7 | `shelter_certainty` | 0.0 | 1.0 | Confiança visual no abrigo percebido. |
| 8 | `shelter_occluded` | 0.0 | 1.0 | 1 se o abrigo está em alcance visual, mas ocluído. |
| 9 | `shelter_dx` | -1.0 | 1.0 | Deslocamento horizontal relativo do abrigo detectado. |
| 10 | `shelter_dy` | -1.0 | 1.0 | Deslocamento vertical relativo do abrigo detectado. |
| 11 | `predator_visible` | -1.0 | 1.0 | 1 se o predador está visível com confiança suficiente. |
| 12 | `predator_certainty` | 0.0 | 1.0 | Confiança visual no predador percebido. |
| 13 | `predator_occluded` | 0.0 | 1.0 | 1 se o predador está em alcance visual, mas ocluído. |
| 14 | `predator_dx` | -1.0 | 1.0 | Deslocamento horizontal relativo do predador detectado. |
| 15 | `predator_dy` | -1.0 | 1.0 | Deslocamento vertical relativo do predador detectado. |
| 16 | `day` | -1.0 | 1.0 | 1 durante o dia. |
| 17 | `night` | -1.0 | 1.0 | 1 durante a noite. |

## `sensory_cortex`

- Observation key: `sensory`
- Role: `proposal`
- Version: `1`
- Description: Propositor sensorial que integra dor, estado corporal, cheiro e luminosidade.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `recent_pain` | 0.0 | 1.0 | Dor recente causada por ataque do predador. |
| 2 | `recent_contact` | 0.0 | 1.0 | Contato físico recente com o predador. |
| 3 | `health` | 0.0 | 1.0 | Saúde corporal. |
| 4 | `hunger` | 0.0 | 1.0 | Fome corporal. |
| 5 | `fatigue` | 0.0 | 1.0 | Fadiga corporal. |
| 6 | `food_smell_strength` | 0.0 | 1.0 | Intensidade do cheiro de alimento. |
| 7 | `food_smell_dx` | -1.0 | 1.0 | Gradiente horizontal do cheiro de alimento. |
| 8 | `food_smell_dy` | -1.0 | 1.0 | Gradiente vertical do cheiro de alimento. |
| 9 | `predator_smell_strength` | 0.0 | 1.0 | Intensidade do cheiro do predador. |
| 10 | `predator_smell_dx` | -1.0 | 1.0 | Gradiente horizontal do cheiro do predador. |
| 11 | `predator_smell_dy` | -1.0 | 1.0 | Gradiente vertical do cheiro do predador. |
| 12 | `light` | 0.0 | 1.0 | Nível simples de luminosidade. |

## `hunger_center`

- Observation key: `hunger`
- Role: `proposal`
- Version: `1`
- Description: Propositor homeostático voltado a forrageio e retorno à última comida percebida.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `hunger` | 0.0 | 1.0 | Estado interno de fome. |
| 2 | `on_food` | 0.0 | 1.0 | 1 se a aranha está sobre alimento. |
| 3 | `food_visible` | 0.0 | 1.0 | 1 se há alimento visível. |
| 4 | `food_certainty` | 0.0 | 1.0 | Confiança visual no alimento percebido. |
| 5 | `food_occluded` | 0.0 | 1.0 | 1 se há alimento próximo, mas ocluído. |
| 6 | `food_dx` | -1.0 | 1.0 | Direção horizontal do alimento detectado. |
| 7 | `food_dy` | -1.0 | 1.0 | Direção vertical do alimento detectado. |
| 8 | `food_smell_strength` | 0.0 | 1.0 | Intensidade do cheiro do alimento. |
| 9 | `food_smell_dx` | -1.0 | 1.0 | Gradiente horizontal do cheiro do alimento. |
| 10 | `food_smell_dy` | -1.0 | 1.0 | Gradiente vertical do cheiro do alimento. |
| 11 | `food_memory_dx` | -1.0 | 1.0 | Direção horizontal da última comida vista. |
| 12 | `food_memory_dy` | -1.0 | 1.0 | Direção vertical da última comida vista. |
| 13 | `food_memory_age` | 0.0 | 1.0 | Idade normalizada da memória de comida. |

## `sleep_center`

- Observation key: `sleep`
- Role: `proposal`
- Version: `1`
- Description: Propositor homeostático voltado a retorno ao abrigo, repouso e profundidade segura.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `fatigue` | 0.0 | 1.0 | Estado interno de fadiga. |
| 2 | `hunger` | 0.0 | 1.0 | Estado interno de fome. |
| 3 | `on_shelter` | 0.0 | 1.0 | 1 se a aranha está no abrigo. |
| 4 | `night` | 0.0 | 1.0 | 1 durante a noite. |
| 5 | `home_dx` | -1.0 | 1.0 | Vetor interno horizontal até o abrigo. |
| 6 | `home_dy` | -1.0 | 1.0 | Vetor interno vertical até o abrigo. |
| 7 | `home_dist` | 0.0 | 1.0 | Distância normalizada até o abrigo. |
| 8 | `health` | 0.0 | 1.0 | Saúde corporal. |
| 9 | `recent_pain` | 0.0 | 1.0 | Dor recente. |
| 10 | `sleep_phase_level` | 0.0 | 1.0 | Nível atual da fase de sono. |
| 11 | `rest_streak_norm` | 0.0 | 1.0 | Continuidade recente do repouso. |
| 12 | `sleep_debt` | 0.0 | 1.0 | Dívida acumulada de sono. |
| 13 | `shelter_role_level` | 0.0 | 1.0 | Profundidade atual no abrigo. |
| 14 | `shelter_memory_dx` | -1.0 | 1.0 | Direção horizontal do abrigo seguro memorizado. |
| 15 | `shelter_memory_dy` | -1.0 | 1.0 | Direção vertical do abrigo seguro memorizado. |
| 16 | `shelter_memory_age` | 0.0 | 1.0 | Idade normalizada da memória do abrigo. |

## `alert_center`

- Observation key: `alert`
- Role: `proposal`
- Version: `1`
- Description: Propositor defensivo voltado a ameaça, fuga e priorização do abrigo sob risco.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `predator_visible` | 0.0 | 1.0 | 1 se o predador está visível. |
| 2 | `predator_certainty` | 0.0 | 1.0 | Confiança visual no predador percebido. |
| 3 | `predator_occluded` | 0.0 | 1.0 | 1 se o predador está em alcance, mas ocluído. |
| 4 | `predator_dx` | -1.0 | 1.0 | Direção horizontal do predador detectado. |
| 5 | `predator_dy` | -1.0 | 1.0 | Direção vertical do predador detectado. |
| 6 | `predator_dist` | 0.0 | 1.0 | Distância normalizada ao predador. |
| 7 | `predator_smell_strength` | 0.0 | 1.0 | Intensidade do cheiro do predador. |
| 8 | `home_dx` | -1.0 | 1.0 | Vetor interno horizontal até o abrigo. |
| 9 | `home_dy` | -1.0 | 1.0 | Vetor interno vertical até o abrigo. |
| 10 | `recent_pain` | 0.0 | 1.0 | Dor recente. |
| 11 | `recent_contact` | 0.0 | 1.0 | Contato físico recente. |
| 12 | `on_shelter` | 0.0 | 1.0 | 1 se a aranha está no abrigo. |
| 13 | `night` | 0.0 | 1.0 | 1 durante a noite. |
| 14 | `predator_memory_dx` | -1.0 | 1.0 | Direção horizontal da última posição vista do predador. |
| 15 | `predator_memory_dy` | -1.0 | 1.0 | Direção vertical da última posição vista do predador. |
| 16 | `predator_memory_age` | 0.0 | 1.0 | Idade normalizada da memória do predador. |
| 17 | `escape_memory_dx` | -1.0 | 1.0 | Direção horizontal da rota recente de fuga. |
| 18 | `escape_memory_dy` | -1.0 | 1.0 | Direção vertical da rota recente de fuga. |
| 19 | `escape_memory_age` | 0.0 | 1.0 | Idade normalizada da memória de fuga. |

## `action_center_context`

- Observation key: `action_context`
- Role: `context`
- Version: `1`
- Description: Contexto bruto usado pelo action_center para arbitrar propostas locomotoras concorrentes.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `hunger` | 0.0 | 1.0 | Fome corporal. |
| 2 | `fatigue` | 0.0 | 1.0 | Fadiga corporal. |
| 3 | `health` | 0.0 | 1.0 | Saúde corporal. |
| 4 | `recent_pain` | 0.0 | 1.0 | Dor recente. |
| 5 | `recent_contact` | 0.0 | 1.0 | Contato recente com o predador. |
| 6 | `on_food` | 0.0 | 1.0 | 1 se está sobre comida. |
| 7 | `on_shelter` | 0.0 | 1.0 | 1 se está sobre o abrigo. |
| 8 | `predator_visible` | 0.0 | 1.0 | 1 se o predador está visível. |
| 9 | `predator_certainty` | 0.0 | 1.0 | Confiança visual no predador. |
| 10 | `predator_dist` | 0.0 | 1.0 | Distância normalizada ao predador. |
| 11 | `day` | 0.0 | 1.0 | 1 durante o dia. |
| 12 | `night` | 0.0 | 1.0 | 1 durante a noite. |
| 13 | `last_move_dx` | -1.0 | 1.0 | Último deslocamento horizontal realizado. |
| 14 | `last_move_dy` | -1.0 | 1.0 | Último deslocamento vertical realizado. |
| 15 | `sleep_debt` | 0.0 | 1.0 | Dívida acumulada de sono. |
| 16 | `shelter_role_level` | 0.0 | 1.0 | Profundidade atual no abrigo. |

## `motor_cortex_context`

- Observation key: `motor_context`
- Role: `context`
- Version: `1`
- Description: Contexto bruto usado pelo motor_cortex para corrigir e executar a intenção locomotora.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `on_food` | 0.0 | 1.0 | 1 se está sobre comida. |
| 2 | `on_shelter` | 0.0 | 1.0 | 1 se está sobre o abrigo. |
| 3 | `predator_visible` | 0.0 | 1.0 | 1 se o predador está visível. |
| 4 | `predator_certainty` | 0.0 | 1.0 | Confiança visual no predador. |
| 5 | `predator_dist` | 0.0 | 1.0 | Distância normalizada ao predador. |
| 6 | `day` | 0.0 | 1.0 | 1 durante o dia. |
| 7 | `night` | 0.0 | 1.0 | 1 durante a noite. |
| 8 | `last_move_dx` | -1.0 | 1.0 | Último deslocamento horizontal realizado. |
| 9 | `last_move_dy` | -1.0 | 1.0 | Último deslocamento vertical realizado. |
| 10 | `shelter_role_level` | 0.0 | 1.0 | Profundidade atual no abrigo. |
