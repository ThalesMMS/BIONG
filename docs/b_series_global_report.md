# Relatorio Global Da Evolucao B-Series

Atualizado em 2026-05-17.

Este relatorio resume a evolucao aceita da serie B, de B0 ate B62, olhando a
linha como um programa incremental de transferencia: cada nivel aceito parte do
checkpoint `best` do nivel anterior, preserva o action space publico primitivo
de `SpiderWorld.step()`, mantem as mesmas seis acoes semanticas internas e
adiciona uma pequena camada de capacidade, modularidade ou bioinspiracao.

A regra central da serie B nao foi "superar sempre o nivel anterior". O criterio
consolidado foi: reter aprendizado/sobrevivencia razoavel, manter os contratos
publicos do mundo, registrar a nova camada em trace/metadata e aceitar apenas
candidatos que passem o gate definido para aquela fase. Candidatos que falham
ficam documentados em `discarded` e nao viram fonte de transferencia.

## 1. Bioinspiracao

A evolucao B comecou de uma observacao biologicamente simples: a aranha antiga
aprendia quando podia escolher intencoes ecologicas como procurar comida,
voltar ao abrigo, ficar, comer ou dormir. B0 preservou essa ideia como
representacao interna, mas retirou o privilegio do ambiente: o mundo atual
continua recebendo somente acoes primitivas.

De B0 a B5, a bioinspiracao e principalmente etologica e homeostatica:

| Faixa | Ideia biologica dominante | Papel na aranha |
| --- | --- | --- |
| B0 | Intencao ecologica interna | Separar "o que fazer" de "qual movimento primitivo executar". |
| B1 | Guarda de ameaca | Impedir que a busca por comida ignore sinais basicos de predador. |
| B2 | Ameaca temporal | Usar memoria curta de ameaca para nao sair cedo demais do abrigo. |
| B3 | Memoria de contato e guarda recorrente | Tornar dor/contato recente relevantes depois do tick imediato. |
| B4 | Recuperacao multi-episodio | Reequilibrar sono, saude e saida de abrigo em episodios variados. |
| B5 | Arbitro homeostatico | Separar fome, sono, recuperacao e ameaca em pressoes interoceptivas. |

De B6 a B14, o foco biologico migrou para risco espacial e affordances. A
aranha passou a registrar se o corredor era viavel, se havia orcamento corporal
para atravessar, se deveria abortar/retornar e se a decisao vinha de um mapa
local, waypoint, replay prospectivo, confianca, atencao preditiva, busca local
ou incerteza. Essa faixa nao resolveu necessariamente todo o corredor, mas
criou sinais auditaveis de decisao: continuar, abortar, retornar, procurar
localmente ou preservar a politica herdada.

De B15 a B22, a inspiracao virou mais cognitiva:

- B15/B16 introduziram opcoes e ensemble de opcoes, aproximando a acao de uma
  escolha hierarquica.
- B17 modulou essas opcoes por sinais neuromodulatorios.
- B18/B19 adicionaram tracos de elegibilidade e memoria episodica.
- B20 formalizou um gate de memoria de trabalho.
- B21/B22 adicionaram replay hipocampal e replay prospectivo de mapas.

De B23 a B39, a serie passou por uma camada de monitoramento, valor e
representacao interna:

- B23/B24 monitoram conflito e precisao do conflito.
- B25/B26 adicionam confianca metacognitiva e predicao alostatica.
- B27/B29 modulam ganho, atencao interoceptiva e competicao de saliencia.
- B30/B33 incorporam gate de ganglios da base, dopamina, ator-critico e erro
  temporal-difference.
- B34/B39 adicionam credito temporal, modelo preditivo, estado latente,
  fatoracao de estado, atencao por fator e binding atencional.

De B40 a B48, a bioinspiracao se aproximou de circuitos cortico-talamicos e
cerebelares:

- B40 cria um global workspace leve.
- B41 adiciona workspace executivo.
- B42/B43 monitoram erro e ajustam precisao adaptativa.
- B44/B46 modelam relay talamico, inibicao reticular e feedback
  corticotalamico.
- B47 adiciona sincronizacao oscilatoria.
- B48 introduz temporizacao cerebelar.

De B49 a B62, a linha ficou explicitamente neuromodulatoria e subcortical:

- B49/B50: gate estriatal e chunking de habito.
- B51/B54: modulacao dopaminergica, precisao colinergica, arousal
  noradrenergico e paciencia serotonergica.
- B55/B57: acoplamento hipotalamico, eixo HPA e consciencia interoceptiva
  insular.
- B58/B60: monitor ACC, contexto prefrontal e valor orbitofrontal.
- B61/B62: valor de seguranca/ameaca amigdalar e seletor defensivo tipo PAG
  para modos `safe_advance`, `flee_to_shelter` e `freeze_hold`.

O resultado global e uma pilha bioinspirada de controle: primeiro sobrevivencia
homeostatica, depois risco espacial, depois memoria/planejamento, depois
metacognicao/valor, depois broadcast/relay cortical e, por fim, modulacao
subcortical de drive, stress, seguranca e defesa.

## 2. Neuromodularidade

A serie B modularizou o comportamento sem modularizar o ambiente. O contrato
externo permaneceu constante: `SpiderWorld.step()` recebe somente uma das nove
acoes primitivas de `ACTIONS`; as seis acoes semanticas da serie B continuam
internas e sao convertidas pela ponte semantica-primitiva.

Essa escolha criou uma modularidade em camadas:

1. **Modulo semantico interno**: escolhe uma intencao auditavel como
   `MOVE_TO_FOOD`, `MOVE_TO_SHELTER`, `EXPLORE`, `STAY`, `EAT` ou `SLEEP`.
2. **Modulo ponte**: transforma a intencao em movimento primitivo, usando
   sinais locais de comida, abrigo, bloqueio e transicao.
3. **Modulo controlador Bx**: em cada nivel B, adiciona um novo conjunto de
   sinais, locks, balances e decisoes.
4. **Modulo de transferencia**: inicializa Bx a partir do cerebro aceito de
   B(x-1), registrando coverage, parent, target, source checkpoint e chaves
   carregadas.
5. **Modulo de validacao**: aceita ou descarta candidatos com gates focados em
   retencao, probes e evidencias de trace.

O padrao de cada modulo Bx se estabilizou em torno de campos de trace:

- `bX_controller_profile`: qual perfil/controlador esta ativo.
- sinais principais do modulo: por exemplo valor, conflito, atencao, memoria,
  drive, stress, seguranca ou defesa.
- um sinal de composicao: balance, confidence, stability, salience ou pressure.
- um lock episodico/transitorio: preserva commits sem transformar a acao em
  macro acao publica.
- `bX_decision`: decisao explicita que permite rejeitar clones do nivel
  anterior.

Essa forma de modularidade e importante porque evita que a serie B vire uma
lista de nomes biologicos sem efeito observavel. Cada modulo precisa deixar
evidencia mensuravel no trace e precisa manter o contrato primitivo.

A progressao modular pode ser lida em blocos:

| Bloco | Modulos principais | Ganho modular |
| --- | --- | --- |
| B0-B5 | ponte, ameaca, contato, recuperacao, homeostase | Modularidade minima entre intencao, ponte e drive corporal. |
| B6-B14 | risco-corredor, recorrencia, affordance, mapa, waypoint, replay, confianca, atencao | Separacao entre risco espacial, orcamento corporal e planejamento local. |
| B15-B22 | opcoes, ensembles, elegibilidade, memoria episodica, working memory, replay hipocampal | Modularidade temporal e hierarquica de decisao. |
| B23-B39 | conflito, precisao, metacognicao, valor, estado latente, fatoracao, binding | Modularidade representacional e de monitoramento. |
| B40-B48 | workspace, relay/inibicao/feedback talamico, sincronizacao, temporizacao | Modularidade de integracao global e coordenacao temporal. |
| B49-B62 | estriado, habito, neuromoduladores, hipotalamo, HPA, insula, ACC, PFC, OFC, amigdala, defesa | Modularidade subcortical e afetivo-homeostatica. |

Do ponto de vista de engenharia, a neuromodularidade da serie B e incremental e
auditavel: o modulo novo nao substitui o anterior; ele o envolve, le os sinais
herdados e decide se preserva, intensifica, aborta, retorna ou bloqueia uma
decisao.

## 3. Capacidade Da Rede

A capacidade da rede aumentou de forma conservadora. B0 estabeleceu a ponte no
mundo atual. B1 introduziu `b_hidden_dim=48` e transferencia parcial de B0,
com coverage aproximado de `0.666896`. A partir dai, a linha aceita quase
sempre promoveu a variante `h48`, porque ela preservou o comportamento com
coverage `1.0` na maioria das transicoes. Variantes `h56` foram usadas como
fallback de capacidade leve, geralmente com coverage esperado em torno de
`0.857`, mas raramente precisaram ser promovidas.

Isso mostra que a evolucao da serie B nao foi principalmente "aumentar o
numero de neuronios". O ganho de capacidade veio mais de tres fontes:

- **Estado interno adicional**: cada nivel adicionou memoria curta, locks,
  acumuladores ou sinais decaidos.
- **Decomposicao funcional**: fome, sono, ameaca, seguranca, confianca,
  conflito, valor, atencao e defesa passaram a ter canais separados.
- **Validacao ambiental progressiva**: a mesma rede pequena precisou reter
  easy/canonical e depois responder a probes como `food_deprivation`,
  `sleep_vs_exploration_conflict`, `food_vs_predator_conflict` e
  `corridor_gauntlet`.

O crescimento de capacidade pode ser resumido assim:

| Etapa | Tipo de capacidade adicionada | Exemplo |
| --- | --- | --- |
| B0 | Capacidade de interface | Converter intencao semantica interna em acao primitiva valida. |
| B1-B5 | Capacidade homeostatica e temporal | Guardas de ameaca, memoria de contato, sono/fome/recuperacao. |
| B6-B14 | Capacidade espacial e prospectiva | Risco de corredor, mapa local, waypoint, replay e atencao preditiva. |
| B15-B22 | Capacidade hierarquica e episodica | Opcoes, ensembles, elegibilidade, memoria de trabalho e replay. |
| B23-B39 | Capacidade avaliativa e representacional | Conflito, precisao, valor, estado latente, fatores e binding. |
| B40-B48 | Capacidade integrativa | Workspace, relay talamico, feedback, sincronizacao e timing. |
| B49-B62 | Capacidade neuromodulatoria e defensiva | Habito, dopamina, colina, noradrenalina, serotonina, HPA, amigdala e PAG. |

O padrao de resultados tambem sugere que a arquitetura `h48` e suficiente para
reter os comportamentos aceitos quando o novo nivel e uma camada fina sobre o
anterior. Quando a mudanca exigiu busca parametrica, a capacidade foi explorada
por perfis fixos e genetic search, nao por inflar indiscriminadamente a rede.

## 4. Tecnicas De Redes Neurais E Complexidade Da Rede

A tecnica central da serie B e transfer learning incremental. B1 e posteriores
exigem `b_parent_level`, `b_transfer_source_checkpoint`,
`b_transfer_min_coverage` e `b_transfer_allow_low_coverage=False`. A carga e
feita por nomes de parametros estaveis e sobreposicao de shape compativel:
parametros compativeis sao carregados; parametros novos ou incompativeis ficam
inicializados normalmente; tudo e registrado em metadata.

Tecnicas usadas ao longo da serie:

- **Transfer learning por checkpoint aceito**: Bx parte do `best` de B(x-1),
  nao de candidatos descartados.
- **Partial weight loading**: permite aumentar de `h48` para `h56` sem exigir
  shapes identicos, desde que a coverage minima seja respeitada.
- **Controladores com estado recorrente leve**: locks, memorias decaidas,
  valores acumulados e cooldowns episodicos.
- **Gates deterministas de promocao**: candidatos fixos podem rodar em
  paralelo, mas a promocao segue prioridade definida, nao ordem de conclusao.
- **Busca genetica local**: usada quando thresholds interdependentes eram o
  gargalo, como em B4/B5/B6 e nos fallbacks parametrizados.
- **Ensembles e opcoes**: B15-B17 introduzem decisao por opcoes e modulacao
  de ensemble.
- **Credito temporal e RL value signals**: B18, B31-B34 adicionam tracos de
  elegibilidade, dopamina, ator-critico, TD error e credito atrasado.
- **Model-based/predictive components**: B10-B12, B21-B22 e B35-B36 adicionam
  replay, predicao, modelo de transicao e estado latente.
- **Attention/workspace mechanisms**: B12-B14, B38-B41 e B43 usam atencao,
  binding, workspace e precisao adaptativa.
- **Neuromodulatory gain control**: B27, B43, B51-B54 e B56 modulam thresholds,
  paciencia, arousal, precisao, stress e recuperacao.

A complexidade da rede cresceu em camadas, nao em largura bruta. Cada nova
fase adicionou:

1. constantes de politica e checkpoint;
2. config no catalogo com parent/source/coverage;
3. runtime que envolve a decisao do nivel anterior;
4. campos novos em `BrainStep` e trace;
5. gate que rejeita clone do nivel anterior sem evidencia do novo modulo;
6. runner que salva `attempt_report.json`, `metadata.json`, `best` ou
   `discarded`;
7. testes unitarios de catalogo, transferencia, runtime, trace e gate.

Ao final de B62, a linha aceita contem uma pilha grande de tecnicas, mas o
contrato operacional continua estreito:

- action space publico: ainda primitivo;
- acoes semanticas internas: ainda as mesmas seis;
- checkpoint fonte: sempre o nivel aceito imediatamente anterior;
- rede aceita dominante: `h48`;
- coverage tipica recente: `1.0`;
- validacao recorrente: compile, testes B-series, testes GUI e `git diff --check`;
- comportamento aceito recente: easy `0..4`, canonical `0..9`, probes
  homeostaticos, conflito comida-predador e evidencia de corredor.

A fase atual aceita e B62:

`artifacts/b_series/evolution/b62_defensive_mode_selector_h48_bridge_policy/seed_7/best`

Ela representa uma arquitetura neural ainda pequena em largura, mas altamente
modular em controle: a decisao final passa por intencao semantica, ponte
primitiva, homeostase, risco, memoria, planejamento, metacognicao, workspace,
neuromodulacao, valor afetivo e modo defensivo antes de chegar a uma unica
acao primitiva enviada ao mundo.
