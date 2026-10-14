# Otimização do custo de um tanque cilíndrico

Trabalho computacional da disciplina **Modelagem Computacional** (IPRJ01-10760 / IPRJ01-11873) - UERJ/IPRJ, Prof. Gustavo Barbosa Libotte.

Aluno: Augusto Ricardo Bonow Moraes dos Santos

## Problema

Encontrar o raio `r` e a altura `h` que minimizam o custo de fabricação de um tanque cilíndrico com volume mínimo de 500 m³:

```
min  f(r,h) = 350·π·r² + 200·π·r·h            (custo em R$)
s.a. g(r,h) = 500 − π·r²·h ≤ 0
```

## Método

1. **Penalização exterior:** `P(r,h) = f(r,h) + μ·[max(0, g(r,h))]²`
2. **Derivadas por diferenças finitas adiantadas:** `∂P/∂r ≈ [P(r+Δ,h) − P(r,h)]/Δ` (idem para `h`)
3. **Método do gradiente com passo fixo:** `x(k+1) = x(k) − α·∇P(x(k))`

Critérios de parada: `‖∇P‖ < 10⁻³`, `‖x(k+1) − x(k)‖ < 10⁻¹²` ou número máximo de iterações. A divergência numérica também é detectada.

## Como executar

```bash
pip install -r requirements.txt
python otimizacao_tanque.py            # todos os experimentos (~1 min)
python otimizacao_tanque.py --rapido   # versão reduzida
```

Os resultados (figuras `.png`, tabelas `.csv`, `resultados.json` e log) são gravados na pasta `resultados/`.

## Experimentos realizados

| Experimento | Descrição |
|---|---|
| Caso base | μ = 10², α = 10⁻⁷, Δ = 10⁻⁵, chute inicial (1, 1) |
| Teste de α | α de 10⁻³ a 10⁻⁸ com μ = 10² |
| Teste de μ | μ de 10¹ a 10⁶ com α = 10⁻⁵/μ |
| Mapa μ × α | combinações para identificar estabilidade/divergência |
| Teste de Δ | Δ de 10⁻³ a 10⁻⁹ |
| Penalização sequencial | μ = 10 → 10⁴ com partida a quente |

## Resultado de referência (analítico)

r* = 3,5693 m, h* = 12,4926 m (h/r = 3,5), custo mínimo f* = R$ 42.024,95.
