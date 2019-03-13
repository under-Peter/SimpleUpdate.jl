struct SimpleUpdateProblem{TU,TA,TW}
    u::TU
    a::TA
    b::TA
    σab::TW
    σba::TW
    ωab::TW
    ωba::TW
end

"""
    simpleupdate(u, ainit, binit)
creates an object of type `SimpleUpdateProblem` that contains the propagator `u`,
the intial tensors `a` and `b` as well as the weights σab, σba, ωab and ωba.
Graphically the structure looks like:

          |_________|
          [____u____]
          |         |
          |  ωab    | ωba
          | /       | /
    -σba-[a]--σab--[b]-σba-
         /         /
       ωba       ωab

with the indices-ordering like:

    3|________4|
     [____u____]
    1|        2|


       5| /3
    4--[a]--2
       /1

"""
function simpleupdate(u, ainit, binit)
    a, b, σab, σba, ωab, ωba = extract_weights!(deepcopy(ainit),deepcopy(binit))
    apply!.((σab, σba, ωab, ωba), x -> x .= x ./ norm(x))
    return SimpleUpdateProblem{typeof(u), typeof(a), typeof(σab)}(u, a, b, σab, σba, ωab, ωba)
end

"""
    extract_weight!(a,b)
returns `σ`, `a'`, `b'` where `σ` is a weight on the leg connecting index 2 of `a`
with index 4 of `b` (i.e. `a` is to the left of `b`).
The weight is extracted by the following steps:
1. input


       | /  | /
    --[a]--[b]--
      /    /

2. isommetrize w.r.t connecting leg


        | /                          | /
     --[Ua]--[Σa]--[Va]-[Ub]--[Σb]--[b]--
       /                            /

3. svd on leg-matrix


     -[c]- = --[Σa]--[Va]-[Ub]--[Σb]--

     -[Uc]--[σc]--[Vc]- = -[c]-

4. reabsorbing unitaries


         | /           | /
     --[a*Uc]--[σc]--[Vc*b]--
        /             /
"""
function extract_weight!(a,b)
    Ua, Σa, Vda = tensorsvd(a, ((1,3,4,5),(2,)))
    Ub, Σb, Vdb = tensorsvd(b, ((4,),(1,2,3,5)))
    @tensor c[1,2] := Σa[1,-1] * Vda[-1,-2] * Ub[-2,-3] * Σb[-3,2]
    Uc, Σc, Vdc = tensorsvd(c)

    @tensor begin
        a[1,2,3,4,5] = Ua[1,3,4,5,-2] * Uc[-2,2]
        b[1,2,3,4,5] = Vdc[4,-4] * Vdb[-4,1,2,3,5]
    end
    return Σc, a, b
end

"""
    extract_weights!(a,b)
returns  new tensors `a`, `b` and their weights `σab`, `σba`, `ωab`, `ωba`,
where `σ` are horizontal and `ω` vertical weights and `ab`/`ba` is read from
left to right and down to up respectively, i.e.

          |  ωab
          | /
    -σba-[a]--σab--[b]
         /
       ωba
"""
function extract_weights!(a,b)
    σab, a , b = extract_weight!(a,b)
    σba, b , a = extract_weight!(b,a)
    @tensor begin
        arot[1,2,3,4,5] := a[4,1,2,3,5]
        brot[1,2,3,4,5] := b[4,1,2,3,5]
    end

    ωab, arot, brot = extract_weight!(arot, brot)
    ωba, brot, arot = extract_weight!(brot, arot)

    @tensor begin
        a[1,2,3,4,5] = arot[2,3,4,1,5]
        b[1,2,3,4,5] = brot[2,3,4,1,5]
    end

    return a, b, σab, σba, ωab, ωba
end

update(su::SimpleUpdateProblem) = update!(deepcopy(su))

function update!(su::SimpleUpdateProblem)
    @unpack a, b, σab, σba, ωab, ωba, u = su

    update!((a,b,σab), (σba, ωab, ωba, u))
    update!((b,a,σba), (σab, ωba, ωab, u))

    @tensor begin
        arot[1,2,3,4,5] := a[4,1,2,3,5]
        brot[1,2,3,4,5] := b[4,1,2,3,5]
    end

    update!((arot, brot, ωab), (ωba, σab, σba, u))
    update!((brot, arot, ωba), (ωab, σba, σab, u))

    @tensor begin
        a[1,2,3,4,5] = arot[2,3,4,1,5]
        b[1,2,3,4,5] = brot[2,3,4,1,5]
    end

    return su
end

function update!((a,b,σab), (σba, ωab, ωba, u))
    #contract with weights
    @tensor begin
        aw[1,2,3,4,5] := σba[4,-4] * a[-1,2,-3,-4,5] * ωba[1,-1] * ωab[-3,3]
        bw[1,2,3,4,5] := σba[-2,2] * b[-1,-2,-3,4,5] * ωab[1,-1] * ωba[-3,3]
    end

    #reduce tensors and apply u
    qa, ra = tensorqr(aw, ((1,3,4),(5,2)))
    rb, qb = tensorrq(bw, ((4,5),(1,2,3)))

    @tensor θ[1,2,3,4] := ra[1,p1,-1] * σab[-1,-2] * rb[-2,p2,3] * u[p1,p2,2,4]

    #truncate bond dim
    s = a isa DTensor ? size(a,2) : sizes(a,2)[1]
    Uθ, Σθ, Vdθ = tensorsvd(θ, ((1,2),(3,4)), svdtrunc = svdtrunc_maxχ(s))

    @tensor begin
        aw[1,2,3,4,5] = qa[1,3,4,-1] * Uθ[-1,5,2]
        bw[1,2,3,4,5] = Vdθ[4,-4,5] * qb[-4,1,2,3]
    end

    #reextract weights
    iσab, iσba, iωab, iωba = pinv.((σab, σba, ωab, ωba))

    @tensor begin
        a[1,2,3,4,5] = iσba[4,-4] * aw[-1,2,-3,-4,5] * iωba[1,-1] * iωab[-3,3]
        b[1,2,3,4,5] = iσba[-2,2] * bw[-1,-2,-3,4,5] * iωab[1,-1] * iωba[-3,3]
    end

     apply!(Σθ, x -> x .= x ./ sum(diag(x)))
     copyto!(σab, Σθ)
     return a, b, σab
end

"""
    energy(su::SimpleUpdateProblem, h)
returns the energy calculated with diagrams such as:

            √ωab      √ωba
            / |       /|
    √σba-[a*]-σab--[b*]√σba
    |    /|   |    /|  |  |
    | √ωba|   | √ωab|  |  |
    |   | |___|___|_|  |  |
    |   | [____h__|_]  |  |
    |   | | √ωab  | |√ωba |
    |   | | /     | | /   |
    √σba|[a]--σab-|[b]√σba-
        |/        |/
      √ωba      √ωab

which might not be the correct approach
"""
function energy(su::SimpleUpdateProblem, h)
    @unpack a, b, σab, σba, ωab, ωba = su
    sqtσab, sqtσba, sqtωab, sqtωba = apply.((σab, σba, ωab, ωba), x -> x .= sqrt(x))

    @tensor aw[1,2,3,4,5] := a[-1,-2,-3,-4,5] * sqtσba[4,-4] * sqtσab[-2,2] *
                             sqtωba[1,-1] * sqtωab[-3,3]
    @tensor bw[1,2,3,4,5] := b[-1,-2,-3,-4,5] * sqtσab[-4,4] * sqtσba[2,-2] *
                             sqtωab[1,-1] * sqtωba[-3,3]
    #E horizontal a-b
    @tensor ehab = aw[-1,l1,-3,-4,1] * conj(aw)[-1,l2,-3,-4,3] *
                   bw[-5,-6,-7,l1,2] * conj(bw)[-5,-6,-7,l2,4] *
                   h[1,2,3,4]
    @tensor nhab = aw[-1,l1,-3,-4,i1] * conj(aw)[-1,l2,-3,-4,i2] *
                   bw[-5,-6,-7,l1,i1] * conj(bw)[-5,-6,-7,l2,i2]
    #E horizontal b-a
    @tensor ehba = bw[-1,l1,-3,-4,1] * conj(bw)[-1,l2,-3,-4,3] *
                   aw[-5,-6,-7,l1,2] * conj(aw)[-5,-6,-7,l2,4] *
                   h[1,2,3,4]
    @tensor nhba = bw[-1,l1,-3,-4,i1] * conj(bw)[-1,l2,-3,-4,i2] *
                   aw[-5,-6,-7,l1,i1] * conj(aw)[-5,-6,-7,l2,i2]
    #E vertical a-b
    @tensor evab = aw[-1,-2,l1,-4,1] * conj(aw)[-1,-2,l2,-4,3] *
                   bw[l1,-6,-7,-8,2] * conj(bw)[l2,-6,-7,-8,4] *
                   h[1,2,3,4]
    @tensor nvab = aw[-1,-2,l1,-4,i1] * conj(aw)[-1,-2,l2,-4,i2] *
                   bw[l1,-6,-7,-8,i1] * conj(bw)[l2,-6,-7,-8,i2]
    #E vertical b-a
    @tensor evba = bw[-1,-2,l1,-4,1] * conj(bw)[-1,-2,l2,-4,3] *
                   aw[l1,-6,-7,-8,2] * conj(aw)[l2,-6,-7,-8,4] *
                   h[1,2,3,4]
    @tensor nvba = bw[-1,-2,l1,-4,i1] * conj(bw)[-1,-2,l2,-4,i2] *
                   aw[l1,-6,-7,-8,i1] * conj(aw)[l2,-6,-7,-8,i2]
    e = ehab/nhab + ehba/nhba + evab/nvab + evba/nvba
    return e/2
end
