# SimpleUpdate

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://under-Peter.github.io/SimpleUpdate.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://under-Peter.github.io/SimpleUpdate.jl/dev)
[![Build Status](https://travis-ci.com/under-Peter/SimpleUpdate.jl.svg?branch=master)](https://travis-ci.com/under-Peter/SimpleUpdate.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/under-Peter/SimpleUpdate.jl?svg=true)](https://ci.appveyor.com/project/under-Peter/SimpleUpdate-jl)
[![Codecov](https://codecov.io/gh/under-Peter/SimpleUpdate.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/under-Peter/SimpleUpdate.jl)

## About
SimpleUpdate.jl is a package that implements the _simplified update_ as originally suggested in
**Accurate Determination of Tensor network State of Quantum Lattice Models in Two Dimensions**
 and succinctly summarized in e.g.
 **Simulation of strongly correlated fermions in two spatial dimensions with fermionic projected entangled-pair states**.

For the algorithm, the peps is assumed is assumed to consist of two rank-4 tensors `a` and `b`
and four diagonal, non-negative matrices -- σab, σba, ωab, ωba --
that form a Peps of this structure:

```
      ωab       ωba       ωab       ωba
       | /       | /       | /       | /
-σba--[a]--σab--[b]--σba--[a]--σab--[b]--σba
       |         |         |         |  
      ωba       ωab       ωba       ωab
       | /       | /       | /       | /
-σab--[b]--σba--[a]--σab--[b]--σba--[a]--σab
       |         |         |         |  
      ωab       ωba       ωab       ωba
       | /       | /       | /       | /
-σba--[a]--σab--[b]--σba--[a]--σab--[b]--σba
       |         |         |         |  
```

where tensors `a` and `b` alternate and the matrices σ and ω live on the links.

To define the problem of a simple update,
we need to provide two tensors `a_init` and `b_init` and an operator `u` with the legs
oriented as
```

    3|________4|
     |____u____|
    1|        2|

       3| /5
    4--[a]--2
        |1

       3| /5
    4--[b]--2
        |1
```
here the 5th index of `a` and `b` is the _physical_ index, indices 1-4 are virtual.
`u` is a rank-4 tensor that acts on tensors `a` and `b` that are linked by a contraction in 4 distinct configurations.

Calling `su = simpleupdate(a_i, b_i, u)` returns a `SimpleUpdateProblem` struct which
contains the three tensors that specify the problem as well as the matrices
-- weights from now on --
 on the virtual bonds.
The weights can be seen as encoding information about the effective environments of each tensor and they are extracted using `extract_weight!`,
see `?extract_weight!` for the algorithm.

With `update!(su)`, the `SimpleUpdateProblem` `su` is advanced one application of the two-body propagator `u`.
Starting with  a cut-out of the peps
```
      ωab       ωba      
       | /       | /     
-σba--[a]--σab--[b]--σba-
       |         |       
      ωba       ωab      
```
we contract the weights outside with the tensors `a` and `b`
```
   | /       | /     
--[a]--σab--[b]--
   |         |       
```

we apply `u` on the _physical legs_ of `a`,`b`
```
       /_________/
      [____u____]
   | /       | /     
--[a]--σab--[b]--
   |         |       
```
Contracting `u`, `a`, `b` and `σab`, we get a tensor `θ`
```
   |_/_______| /     
--[____θ______]--
   |         |       
```
which we can split using an `svd` where we group all left and right indices together respectively to get
```
   | /       | /     
--[Uθ]-[Σθ]-[V†θ]--
   |         |       
```
where we truncate the bond-dimensions during the `svd` to the desired dimension.

inserting resolutions of the identity `𝟙 = σab * σab⁻¹` we can contract
the inverses with `Uθ` and  `V†θ` respectively and recover the weights on the links again:
```
      ωab       ωba
      | /       | /     
σba--[Uθ]-[Σθ]-[V†θ]--σba
      |         |       
      ωba       ωab
```
We can now set `a := Uθ`, `b := V†θ` and `σab := Σθ`
 to get the _updated_ iPEPS after application of _all_ horizontal propagators on links `a-b`.
We repeat the same for all horizontal propagators on links `b-a` and do the same for all _vertical_ links `a-b` and `b-a`.
This corresponds to a trotterized application of the two-body operator on _all_ links of the iPEPS.
(Note that in the package the physical indices are first moved onto the link before `u` is applied since the computational cost can be reduced that way.)


Truncating the tensor with an `svd` in this way is inspired by the same approach in `TEBD`,
 where the same is valid because the `MPS` can be brought into a canonical form.
 For the iPEPS we don't have a canonical form and thus the truncation implies that the environment has the property that it decomposes into a product state.
 This is a stark simplification but the algorithm still works for some systems.

 If `u` is an imaginary time propagator and the algorithm is used for ground-state search with imaginary time evolution,
it does not minimize the global energy of the iPEPS but rather local terms of the form
```
         ωab       ωba
        / |       /|
 σba-[a*]-σab--[b*]-σba
 |   /|   |    /|  |  |
 | ωba|   |  ωab|  |  |
 |  | |___|___|_|  |  |
 |  | [____h__|_]  |  |
 |  | |   |   | |  |  |
 |  | |  ωab  | | ωba |
 |  | | /     | | /   |
 σba|[a]--σab-|[b]-σba-
    |/        |/
   ωba       ωab
```
which are returned by `energy(su, h)`.
To check for convergence, this is the quantity to track.
