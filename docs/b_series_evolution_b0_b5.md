# Evolução B-Series: B0 a B5

Atualizado em 2026-05-16.

Este documento resume apenas a linha B-series que deu certo, isto é, os
checkpoints aceitos e reutilizados como fonte de transferência para o nível
seguinte. Candidatos descartados são citados apenas quando ajudam a explicar por
que a arquitetura aceita foi escolhida.

## Princípios Da Linha B

A linha B-series é uma trilha diagnóstica paralela. Ela não substitui a linha A
por si só. O objetivo de cada nível `Bx` é preservar aprendizado/sobrevivência
razoável enquanto adiciona uma pequena dose de complexidade, modularidade ou
bioinspiração em relação a `B(x-1)`.

As regras que se consolidaram de B0 a B5:

- O action space público de `SpiderWorld.step()` continua primitivo. Nenhuma
  ação semântica entra no mundo.
- As seis ações semânticas continuam internas: `MOVE_TO_FOOD`,
  `MOVE_TO_SHELTER`, `EXPLORE`, `STAY`, `EAT` e `SLEEP`.
- A ponte semântica-primitiva transforma a intenção interna em uma ação de
  `ACTIONS`.
- A partir de B1, todo nível aceito nasce por transfer learning a partir do
  checkpoint `best` aceito do nível anterior.
- Candidato que não passa o gate do nível é salvo em `discarded` e não vira
  substrato evolutivo.
- A validação local é focada em B-series e smoke de comportamento; a suíte
  completa continuar parcialmente vermelha não invalida a promoção de um nível B.

## Cadeia Aceita

| Nível | Arquitetura aceita | Fonte usada | Coverage | Gate principal |
| --- | --- | --- | ---: | --- |
| B0 | `b0_current_bridge_policy` | sem fonte B anterior | n/a | easy completo |
| B1 | `b1_threat_guard_bridge_policy` | B0 current bridge | 0.666896 | easy completo |
| B2 | `b2_temporal_threat_h48_bridge_policy` | B1 threat guard | 1.0 | easy + progresso canonical |
| B3 | `b3_recurrent_guard_h48_bridge_policy` | B2 temporal threat | 1.0 | easy + canonical ep0/ep1 robusto |
| B4 | `b4_genetic_recovery_h48_bridge_policy` | B3 recurrent guard | 1.0 | easy 0..4 + retenção canonical 0..9 |
| B5 | `b5_genetic_homeostasis_h48_bridge_policy` | B4 genetic recovery | 1.0 | easy 0..4 + canonical 0..9 + probes homeostáticos |

Checkpoints aceitos:

- B0:
  `artifacts/b_series/evolution/b0_current_bridge_policy/seed_7/best`
- B1:
  `artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best`
- B2:
  `artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best`
- B3:
  `artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best`
- B4:
  `artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best`
- B5:
  `artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best`

## B0: Ponte Semântica Primitiva Atual

Arquitetura aceita: `b0_current_bridge_policy`.

B0 criou o substrato mínimo da linha atual: a política ainda produz e registra
intenções semânticas internas, mas o mundo recebe somente uma ação primitiva. O
nível separa três responsabilidades:

- seleção semântica auditável;
- ponte de movimento local para comida/abrigo;
- execução primitiva no mundo atual.

A diferença fundamental em relação ao `b0_legacy_semantic_policy` é que B0
current não dá ações macro ao ambiente. `MOVE_TO_FOOD` e `MOVE_TO_SHELTER` não
entram em `SpiderWorld.step()`: elas são convertidas antes em movimento
primitivo.

Resultado registrado:

- seed: `7`
- treino: `24` episódios
- cenário easy: `432` steps vivo
- comida: `21`
- sono: `100`
- entradas em abrigo: `14`
- contatos com predador: `1`
- saúde final: `1.0`
- trace primitivo: válido

B0 é a base evolutiva da linha, mas ainda é deliberadamente simples. A
complexidade principal está na ponte e no ciclo semântico básico, não em um
controle neural mais modular.

## B1: Guardião De Ameaça Inicial

Arquitetura aceita: `b1_threat_guard_bridge_policy`.

Fonte:
`artifacts/b_series/evolution/b0_current_bridge_policy/seed_7/best`.

Transferência:

- parent level: `0`
- target level: `1`
- coverage: `0.666896`
- parâmetros carregados: `6791 / 10183`
- low coverage override: desabilitado

B1 começou como tentativa conservadora de capacidade. Os candidatos
capacity-only `b1_capacity_h48_bridge_policy` e
`b1_capacity_h64_bridge_policy` foram descartados porque não sustentaram o gate
easy. O candidato aceito adicionou uma guarda de ameaça mínima sem alterar o
contrato público do mundo.

Mudança arquitetural principal:

- mantém as seis ações semânticas internas;
- mantém a ponte primitiva;
- aumenta a capacidade oculta para `48`;
- adiciona controle B1-local de ameaça com
  `semantic_action_source="b1_threat_guard_controller"`.

Resultado aceito:

- treino: `24` episódios
- easy: `432` steps vivo
- comida: `21`
- sono: `72`
- entradas em abrigo: `18`
- contatos com predador: `1`
- saúde final: `1.0`

Diagnóstico canonical ainda fraco, usado como referência para B2:

- `69` steps
- comida: `3`
- sono: `3`
- entradas em abrigo: `5`
- contatos com predador: `5`
- saúde final: `0.0`

B1 provou que a transferência de B0 para uma rede maior e uma guarda simples
mantinha o aprendizado no easy, mas ainda não resolvia ameaça temporal no
canonical.

## B2: Controle Temporal De Ameaça

Arquitetura aceita: `b2_temporal_threat_h48_bridge_policy`.

Fonte:
`artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best`.

Transferência:

- parent level: `1`
- target level: `2`
- coverage: `1.0`
- parâmetros carregados: `10183 / 10183`

B2 manteve capacidade `h48`, mesmas ações semânticas e mesma ponte, mas
adicionou um controlador temporal de ameaça. O objetivo foi atacar o bloqueador
do canonical observado em B1: contato e risco de predador não eram tratados com
memória suficiente.

Mudança arquitetural principal:

- `semantic_action_source="b2_temporal_threat_controller"`;
- pressão de ameaça atual;
- memória curta de predador;
- traço temporal de predador;
- tendência a retornar/segurar abrigo quando ameaça recente está ativa.

Resultado aceito:

- treino: `48` episódios
- easy: `432` steps vivo, `19` comidas, `76` eventos de sono, `15` entradas em
  abrigo, `1` contato, saúde final `1.0`
- canonical ep0: `300` steps vivo, `13` comidas, `46` eventos de sono, `12`
  entradas em abrigo, `9` contatos, saúde final `0.063414`

B2 passou porque completou o horizonte canonical e demonstrou progresso claro
contra o baseline B1, embora ainda com muitos contatos.

## B3: Guarda Recorrente E Memória De Contato

Arquitetura aceita: `b3_recurrent_guard_h48_bridge_policy`.

Fonte:
`artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best`.

Transferência:

- parent level: `2`
- target level: `3`
- coverage: `1.0`
- parâmetros carregados: `10183 / 10183`

B3 começou testando memória de contato direta. As variantes
`b3_contact_memory_h48_bridge_policy`,
`b3_contact_memory_strict_h48_bridge_policy` e
`b3_contact_memory_h56_bridge_policy` foram descartadas por falhas em easy ou
canonical. A arquitetura aceita foi a guarda recorrente `h48`, que preservou o
substrato B2 e adicionou perfil transitório por episódio.

Mudança arquitetural principal:

- `semantic_action_source="b3_contact_memory_controller"` na família B3;
- cooldown pós-contato;
- cooldown pós-comida;
- detecção de queda de fome;
- perfil recorrente por episódio, com guarda mais rígida no começo e liberação
  posterior.

Resultado aceito:

- treino: `48` episódios
- easy: `432` steps vivo, `19` comidas, `76` eventos de sono, `15` entradas em
  abrigo, `1` contato, saúde final `1.0`
- canonical ep0: `300` steps vivo, `15` comidas, `39` eventos de sono, `10`
  entradas em abrigo, `2` contatos, saúde final `0.827079`
- canonical ep1: `300` steps vivo, `16` comidas, `65` eventos de sono, `13`
  entradas em abrigo, `1` contato, saúde final `1.0`

B3 foi o primeiro nível a tornar o canonical robusto nos episódios diagnósticos
0 e 1. A melhoria não veio de mais capacidade, mas de memória de estado
transitória e perfil temporal de controle.

## B4: Recuperação Multi-Episódio Com Busca Genética

Arquitetura aceita: `b4_genetic_recovery_h48_bridge_policy`.

Fonte:
`artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best`.

Transferência:

- parent level: `3`
- target level: `4`
- coverage: `1.0`
- parâmetros carregados: `10183 / 10183`

B4 mudou o foco de robustez local ep0/ep1 para retenção multi-episódio em
canonical `0..9`. O objetivo não era obrigatoriamente bater B3 em todos os
números, mas preservar sobrevivência razoável com um módulo de recuperação mais
bioinspirado.

As variantes fixas passaram easy, mas foram descartadas por retenção
insuficiente no canonical multi-episódio. O fallback genético pequeno buscou
thresholds de recuperação e promoveu o primeiro perfil confirmado.

Mudança arquitetural principal:

- `semantic_action_source="b4_genetic_recovery_controller"`;
- pressão de recuperação;
- retenção de sono/abrigo quando saúde, fadiga ou dívida de sono indicam risco;
- bloqueio de saída prematura;
- liberação controlada de forrageamento;
- thresholds vencedores persistidos em `b_controller_params`.

Resultado aceito:

- treino: `48` episódios
- busca: híbrida, workers `4`, população GA `12`, gerações `4`
- easy `0..4`: passou
- canonical `0..9`: passou gate de retenção
- horizontes completos: `4 / 10`
- mínimo de steps: `49`
- total de contatos: `24`
- episódios com ciclo de comida: `9`
- episódios com ciclo de sono: `9`
- episódios com ciclo de abrigo: `10`

Âncoras canonical preservadas:

- ep0: `300` steps vivo, `14` comidas, `54` eventos de sono, `10` entradas em
  abrigo, `2` contatos, saúde final `1.0`
- ep1: `300` steps vivo, `15` comidas, `46` eventos de sono, `11` entradas em
  abrigo, `4` contatos, saúde final `1.0`

Fraqueza residual que motivou B5:

- episódio canonical `8` ainda colapsava por homeostase/saída de abrigo sem
  sono suficiente, perto de `49` steps.

## B5: Árbitro Homeostático Modular

Arquitetura aceita: `b5_genetic_homeostasis_h48_bridge_policy`.

Fonte:
`artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best`.

Transferência:

- parent level: `4`
- target level: `5`
- coverage: `1.0`
- parâmetros carregados: `10183 / 10183`

B5 manteve o checkpoint B4 como substrato e adicionou um árbitro interoceptivo
mais explícito. O nível separou pressões de fome, sono, recuperação e ameaça,
com locks transitórios por episódio.

As variantes fixas passaram easy e os dois probes obrigatórios, mas foram
descartadas por retenção canonical insuficiente. O perfil genético foi aceito.

Mudança arquitetural principal:

- `semantic_action_source="b5_genetic_homeostasis_controller"`;
- `b_effective_level="B5-homeostatic-arbiter"`;
- pressão de fome;
- pressão de sono;
- dívida de recuperação;
- gate de ameaça;
- `sleep_bout_lock`;
- `forage_commitment_lock`;
- decisão homeostática registrada em trace;
- thresholds vencedores persistidos em `b_controller_params`.

Resultado aceito:

- treino: `48` episódios
- busca: híbrida, workers `4`, população GA `12`, gerações `4`
- easy `0..4`: passou
- canonical `0..9`: passou retenção
- horizontes completos: `5 / 10`
- mínimo de steps: `52`
- total de contatos: `24`
- episódios com ciclo de comida: `9`
- episódios com ciclo de sono: `9`
- episódios com ciclo de abrigo: `10`

Probes complexos obrigatórios:

- `food_deprivation` ep0..2: passou, com `2 / 3` episódios de progresso
- `sleep_vs_exploration_conflict` ep0..2: passou, métrica de movimento
  pós-recuperação exposta e `3 / 3` episódios com movimento pós-recuperação

Diagnósticos não bloqueantes ainda problemáticos:

- `food_vs_predator_conflict`: scorers ainda falham em prioridade de ameaça e
  supressão de forrageamento sob ameaça.
- `corridor_gauntlet`: há progresso inicial, mas morte rápida por falta de
  sobrevivência sustentada no corredor.

B5 é, portanto, o primeiro nível da linha a combinar retenção B4, homeostase
modular explícita e probes ambientais adicionais sem alterar o action space
público.

## Leitura Da Evolução

A progressão bem-sucedida não foi uma simples escalada de hidden size. O padrão
que funcionou foi:

1. B0 criou uma ponte auditável entre intenção semântica e ação primitiva.
2. B1 adicionou uma guarda mínima de ameaça quando capacity-only falhou.
3. B2 adicionou memória temporal de ameaça.
4. B3 adicionou memória episódica de contato e perfil recorrente.
5. B4 adicionou recuperação e busca genética sobre thresholds de controle.
6. B5 adicionou arbitragem homeostática modular e probes ambientais mais duros.

Em todos os níveis aceitos, a complexidade nova ficou acima da ponte primitiva,
sem mudar o contrato de `SpiderWorld.step()`.

## Próximos Passos

### 1. Formalizar B6 A Partir Do B5 Aceito

Fonte obrigatória proposta:
`artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best`.

Objetivo de B6: manter a retenção B5 e transformar os diagnósticos não
bloqueantes de B5 em eixos de arquitetura, sem exigir uma superação geral de B5.

Direção recomendada: um módulo bioinspirado de arbitragem risco-forrageamento,
com atenção a ameaça local e affordances de corredor.

Variantes iniciais sugeridas:

- `b6_risk_forage_arbiter_h48_bridge_policy`
- `b6_corridor_affordance_guard_h48_bridge_policy`
- `b6_threat_priority_memory_h48_bridge_policy`
- `b6_genetic_risk_homeostasis_h48_bridge_policy` como fallback se os perfis
  fixos falharem

### 2. Promover Diagnósticos B5 Para Gates B6 Parciais

Os cenários `food_vs_predator_conflict` e `corridor_gauntlet` não devem exigir
sucesso perfeito de início, mas precisam deixar de ser apenas texto diagnóstico.

Gate B6 sugerido:

- manter easy `0..4`;
- manter retenção canonical `0..9` em patamar B5;
- manter `food_deprivation`;
- manter `sleep_vs_exploration_conflict`;
- exigir progresso mensurável em `food_vs_predator_conflict`;
- exigir progresso mensurável em `corridor_gauntlet`.

Critérios parciais possíveis:

- menos contatos ou ameaça priorizada em `food_vs_predator_conflict`;
- mais steps vivos ou morte mais tardia em `corridor_gauntlet`;
- preservar trace primitivo em todos os probes.

### 3. Melhorar Observabilidade Antes De Mais Complexidade

Antes de uma mudança estrutural maior, vale adicionar campos de trace para B6:

- decisão risco-versus-fome;
- ameaça usada pelo arbiter;
- motivo de supressão de forrageamento;
- compromisso de corredor;
- progresso local em corredor;
- motivo de abandono de abrigo;
- lock de retorno ao abrigo.

Isso evita repetir um problema antigo da linha A: ter comportamento ruim sem
conseguir separar falha de percepção, falha de decisão e falha da ponte.

### 4. Preservar O Regime De Descarte

B6 deve continuar descartando candidatos que não passem o gate. O primeiro
candidato aceito vira `best`; descartados não podem virar fonte de transferência.

Documentar para cada candidato:

- fonte B5;
- coverage;
- status `accepted` ou `discarded`;
- motivo de descarte;
- métricas de easy/canonical/probes;
- parâmetros genéticos, quando houver.

### 5. Adiar Multi-Seed Amplo

Ainda não parece hora de transformar B6 em validação multi-seed pesada. O gate
mais útil agora é arquitetura incremental com cenários mais informativos. Depois
que B6 ou B7 estabilizar risco-forrageamento e corredor, a próxima etapa natural
é uma confirmação multi-seed curta.

### 6. Possível Linha B7

Se B6 mantiver B5 e melhorar os diagnósticos de ameaça/corredor, B7 pode ser uma
arquitetura mais estrutural:

- memória recorrente explícita pequena;
- submódulo de affordance local;
- separação mais clara entre interocepção e exterocepção;
- gate com canonical `0..9` e probes de conflito/corredor como obrigatórios.

O critério deve continuar sendo retenção com complexidade modular incremental,
não uma competição direta por maior score bruto.

