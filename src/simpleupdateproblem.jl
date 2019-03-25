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
    a, b = deepcopy(ainit), deepcopy(binit)
    σab = TensorOperations.similar_from_indices(
        real(eltype(a)), (2,), (4,), (1,),(2,), a, b, :Y, :Y)
    σba, ωab, ωba = deepcopy(σab), deepcopy(σab), deepcopy(σab)
    apply!.((σab, σba, ωab, ωba), x -> x .= Matrix(I, size(x)...))
    return SimpleUpdateProblem{typeof(u), typeof(a), typeof(σab)}(u, a, b, σab, σba, ωab, ωba)
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
        a[1,2,3,4,5] = qa[1,3,4,-1] * Uθ[-1,5,2]
        b[1,2,3,4,5] = Vdθ[4,-4,5] * qb[-4,1,2,3]
    end

    #reextract weights
    iσab, iσba, iωab, iωba = pinv.((σab, σba, ωab, ωba))
    @tensor begin
        a[1,2,3,4,5] = a[-1,2,-3,-4,5] * iωba[1,-1] * iωab[3,-3] * iσba[4,-4]
        b[1,2,3,4,5] = b[-1,-2,-3,4,5] * iωab[1,-1] * iωba[3,-3] * iσba[2,-2]
     end

     apply!(Σθ, x -> x .= x ./ norm(x))
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
    @tensor aw[1,2,3,4,5] := a[-1,-2,-3,-4,5] *
                             ωba[1,-1] *
                             σab[-2,2] *
                             ωab[-3,3] *
                             σba[4,-4]

    #E horizontal a-b
    @tensor bw[1,2,3,4,5] := b[-1,-2,-3,4,5] *
                             ωab[1,-1] *
                             σba[2,-2] *
                             ωba[-3,3]
    @tensor ehab = aw[-1,l1,-3,-4,1] * conj(aw)[-1,l2,-3,-4,3] *
                   bw[-5,-6,-7,l1,2] * conj(bw)[-5,-6,-7,l2,4] *
                   h[1,2,3,4]
    @tensor nhab = aw[-1,l1,-3,-4,i1] * conj(aw)[-1,l2,-3,-4,i1] *
                   bw[-5,-6,-7,l1,i2] * conj(bw)[-5,-6,-7,l2,i2]

    #E horizontal b-a
    @tensor bw[1,2,3,4,5] := b[-1,2,-3,-4,5] *
                             ωab[1,-1] *
                             ωba[-3,3] *
                             σab[4,-4]
    @tensor ehba = bw[-1,l1,-3,-4,1] * conj(bw)[-1,l2,-3,-4,3] *
                   aw[-5,-6,-7,l1,2] * conj(aw)[-5,-6,-7,l2,4] *
                   h[1,2,3,4]
    @tensor nhba = bw[-1,l1,-3,-4,i1] * conj(bw)[-1,l2,-3,-4,i1] *
                   aw[-5,-6,-7,l1,i2] * conj(aw)[-5,-6,-7,l2,i2]

    #E vertical a-b
    @tensor bw[1,2,3,4,5] := b[1,-2,-3,-4,5] *
                             σba[2,-2] *
                             ωba[-3,3] *
                             σab[4,-4]
    @tensor evab = aw[-1,-2,l1,-4,1] * conj(aw)[-1,-2,l2,-4,3] *
                   bw[l1,-6,-7,-8,2] * conj(bw)[l2,-6,-7,-8,4] *
                   h[1,2,3,4]
    @tensor nvab = aw[-1,-2,l1,-4,i1] * conj(aw)[-1,-2,l2,-4,i1] *
                   bw[l1,-6,-7,-8,i2] * conj(bw)[l2,-6,-7,-8,i2]

    #E vertical b-a
    @tensor bw[1,2,3,4,5] := b[-1,-2,3,-4,5] *
                             ωab[1,-1] *
                             σba[2,-2] *
                             σab[4,-4]
    @tensor evba = bw[-1,-2,l1,-4,1] * conj(bw)[-1,-2,l2,-4,3] *
                   aw[l1,-6,-7,-8,2] * conj(aw)[l2,-6,-7,-8,4] *
                   h[1,2,3,4]
    @tensor nvba = bw[-1,-2,l1,-4,i1] * conj(bw)[-1,-2,l2,-4,i1] *
                   aw[l1,-6,-7,-8,i2] * conj(aw)[l2,-6,-7,-8,i2]
    es = (ehab/nhab,  ehba/nhba, evab/nvab, evba/nvba)
    return Real.(es)
    return sum(es)/2
end
