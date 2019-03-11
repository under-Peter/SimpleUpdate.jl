using SimpleUpdate
using Test
using TensorOperations, TensorNetworkTensors, LinearAlgebra

@testset "SimpleUpdate.jl" begin
    @testset "Ising" begin
        u = tfisingpropagator(0,1)
        @test toarray(fuselegs(u,((1,2),(3,4)))[1]) ≈ I

        u = tfisingpropagator(1e-3,1)
        @tensor u2[1,2,3,4] := u[1,2,-1,-2] * u[-1,-2,3,4]
        @test u2 ≈ tfisingpropagator(2e-3,1)
        @test ishermitian(toarray(fuselegs(u,((1,2),(3,4)))[1]))
    end

    @testset "Weights" begin
        χ = 4
        a, b = DTensor(rand(χ,χ,χ,χ,2)), DTensor(rand(χ,χ,χ,χ,2))
        #rotate back and forth
        @tensor begin
            arot[1,2,3,4,5] := a[4,1,2,3,5]
            brot[1,2,3,4,5] := b[4,1,2,3,5]
        end
        @tensor begin
            a2[1,2,3,4,5] := arot[2,3,4,1,5]
            b2[1,2,3,4,5] := brot[2,3,4,1,5]
        end
        @test a2 ≈ a
        @test b2 ≈ b

        #[a]--[b] → [a]-c-[b]
        @tensor n1 = a[-1,lab,-2,-3,aa] * conj(a)[-1,lab2,-2,-3,aa] *
                     b[-4,-5,-6,lab,bb] * conj(b)[-4,-5,-6,lab2,bb]
        c, = extract_weight!(a,b)
        @tensor a[1,2,3,4,5] = a[1,-2,3,4,5] * c[-2,2]
        @tensor n2 = a[-1,lab,-2,-3,aa] * conj(a)[-1,lab2,-2,-3,aa] *
                     b[-4,-5,-6,lab,bb] * conj(b)[-4,-5,-6,lab2,bb]
        @test n1 ≈ n2

        #[b]--[a] → [b]-c-[a]
        b, a = DTensor(rand(χ,χ,χ,χ,2)), DTensor(rand(χ,χ,χ,χ,2))
        @tensor n1 = b[-1,lba,-2,-3,bb] * conj(b)[-1,lba2,-2,-3,bb] *
                     a[-4,-5,-6,lba,aa] * conj(a)[-4,-5,-6,lba2,aa]
        c, = extract_weight!(b,a)
        @tensor b[1,2,3,4,5] = b[1,-2,3,4,5] * c[-2,2]
        @tensor n2 = b[-1,lba,-2,-3,bb] * conj(b)[-1,lba2,-2,-3,bb] *
                     a[-4,-5,-6,lba,aa] * conj(a)[-4,-5,-6,lba2,aa]
        @test n1 ≈ n2

        #w/ rotation
        a, b = DTensor(rand(χ,χ,χ,χ,2)), DTensor(rand(χ,χ,χ,χ,2))
        @tensor n1 = a[-1,-2,lab,-3,aa] * conj(a)[-1,-2,lab2,-3,aa] *
                     b[lab,-5,-6,-4,bb] * conj(b)[lab2,-5,-6,-4,bb]
        @tensor begin
            arot[1,2,3,4,5] := a[4,1,2,3,5]
            brot[1,2,3,4,5] := b[4,1,2,3,5]
        end

        c, = extract_weight!(arot, brot)
        @tensor arot[1,2,3,4,5] = arot[1,-2,3,4,5] * c[-2,2]
        @tensor n2 = arot[-1,lab,-2,-3,aa] * conj(arot)[-1,lab2,-2,-3,aa] *
                     brot[-4,-5,-6,lab,bb] * conj(brot)[-4,-5,-6,lab2,bb]
        @test n1 ≈ n2
    end
end
