# Action Center Design Note

## Objetivo

Separar arbitragem comportamental de execução locomotora sem trocar o espaço motor do ambiente.

Nova cadeia:

`módulos especializados ou monolithic_policy -> action_center -> motor_cortex -> ação final`

## Contratos

- `action_center`
  - entrada: concatenação das propostas locomotoras + `action_context`
  - saída: logits locomotores sobre `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`, `STAY`
  - cabeça de valor: sim

- `motor_cortex`
  - entrada: intenção locomotora escolhida pelo `action_center` em one-hot + `motor_context`
  - saída: logits corretivos sobre o mesmo espaço locomotor
  - cabeça de valor: não

## Contextos

- `action_context`
  - estado corporal e situacional amplo para arbitragem: fome, fadiga, saúde, dor, contato, comida/abrigo, predador, ciclo dia/noite, último deslocamento, dívida de sono e profundidade no abrigo

- `motor_context`
  - contexto local de execução/correção: comida/abrigo, predador, ciclo dia/noite, último deslocamento e profundidade no abrigo

## Reflexos

- Os reflexos locais continuam nos módulos propositores.
- Eles modulam propostas antes do `action_center`.
- O `motor_cortex` não aplica reflexos; ele só corrige a execução da intenção arbitrada.

## Monolithic Policy

- `monolithic_policy` continua como baseline de proposta única.
- Ela agora alimenta o mesmo `action_center` e o mesmo `motor_cortex` da arquitetura modular.
- Isso mantém comparabilidade estrutural a jusante da etapa propositora.

## Compatibilidade

- `architecture_signature()` agora inclui `action_center`, `action_context` e o novo `motor_context`.
- `ARCHITECTURE_VERSION` foi incrementada.
- Checkpoints antigos falham com mensagem explícita; não há migração automática nesta entrega.
