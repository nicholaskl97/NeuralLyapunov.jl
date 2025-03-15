using SafeTestsets

@time begin
    @time @safetestset "Simple pendulum" begin
        include("pendulum_test.jl")
    end

    @time @safetestset "Double pendulum" begin
        include("double_pendulum_test.jl")
    end

    @time @safetestset "Planar Quadrotor" begin
        include("planar_quadrotor_test.jl")
    end

    @time @safetestset "Quadrotor (3D)" begin
        include("quadrotor_test.jl")
    end
end
