"""
    tfisinghamiltonian(h)
return the DTensor-representation of the two-body part of the
transverse-field ising Hamiltonian in two dimensions, i.e.

    H = -σx⊗σx - h(𝟙⊗σz + σz⊗𝟙)

Note that depending on the lattice geometry,
`h` needs to be scaled appropriately (e.g. on the square, `h → h/4`).

"""
function tfisinghamiltonian(h, T = ComplexF64)
    σx = T[0 1; 1  0]
    σz = T[1 0; 0 -1]
    id = T[1 0; 0  1]
    @tensor H[1,2,3,4] := -σx[1,3] * σx[2,4] -
                           h * id[1,3] * σz[2,4] -
                           h * id[2,4] * σz[1,3]
    return DTensor(H)
end

"""
    tfisingpropagator(h)
return the rank-4 propagator for a two-site ising-hamlitonian with
transverse field `h`, see `tfisinghamiltonian`.
"""
function tfisingpropagator(β,h)
    H = tfisinghamiltonian(h)
    Hmat, rs = fuselegs(H, ((1,2),(3,4)))
    apply!(Hmat, x -> x .= exp(-β .* x))
    U = splitlegs(Hmat, ((1,1,1),(1,1,2),(2,2,1),(2,2,2)), rs...)
    return U
end
