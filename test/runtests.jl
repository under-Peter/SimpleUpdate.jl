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

end
