# SimpleUpdate

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://under-Peter.github.io/SimpleUpdate.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://under-Peter.github.io/SimpleUpdate.jl/dev)
[![Build Status](https://travis-ci.com/under-Peter/SimpleUpdate.jl.svg?branch=master)](https://travis-ci.com/under-Peter/SimpleUpdate.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/under-Peter/SimpleUpdate.jl?svg=true)](https://ci.appveyor.com/project/under-Peter/SimpleUpdate-jl)
[![Codecov](https://codecov.io/gh/under-Peter/SimpleUpdate.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/under-Peter/SimpleUpdate.jl)

## About
SimpleUpdate.jl is a package that implements the algorithm of the same name for *iPEPS* as follows:

1. The problem is described by providing two initial rank-5 tensors `a` and `b` and a propagator rank-4 tensor `u`  whose indices are ordered as:


    3|________4|
     [____u____]
    1|        2|

       5| /3
    4--[a]--2
       /1
here the 5th index of `a` is the _physical_ index, indices 1-4 are virtual.

2. `simpleupdate(a,b,u)` returns a `SimpleUpdateProblem` struct which contains the three tensors that specify the problem as well as the weights on the virtual bonds that have been extracted to describe an environment.
The weights are extracted using `extract_weight!`, see `?extract_weight!` for the exact algorithm.

3. With `update!(su)`, the `SimpleUpdateProblem` `su` is advanced one application of `u` on each configuration of the two tensors `a` `b`, i.e. horizontally `[a]-[b]` and `[b]-[a]` and rotated by 90°.


## Issues


When doing e.g.
```julia
χ = 4
u = tfisingpropagator(0,0.1)
h = tfisinghamiltonian(0.1)
ainit, binit = DTensor(rand(χ,χ,χ,χ,2)), DTensor(rand(χ,χ,χ,χ,2))
su = simpleupdate(u,ainit,binit)
su2 = update(su)
su3 = update(su2)
energy(su, h)
energy(su2, h)
energy(su3, h)
```
the energy is _not_ constant, despite propagating for `t=0` being a no-op. That could be caused by an error in the code or the energy being calculated wrong.
The terms in the energy are calculated with diagrams such as:
```

            √ωab      √ωba
            / |       /|
    √σba-[a*]-σab--[b*]√σba
    |    /|   |    /|  |  |
    | √ωba|   | √ωab|  |  |
    |   | |   |   | |  |  |
    |   | |___|___|_|  |  |
    |   | [____h__|_]  |  |
    |   | |   |   | |  |  |
    |   | | √ωab  | |√ωba |
    |   | | /     | | /   |
    √σba|[a]--σab-|[b]√σba-
        |/        |/
      √ωba      √ωab

```

which might be wrong (other sources use the tensors `a` and `b` for *CTM*.)
