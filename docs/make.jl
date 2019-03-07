using Documenter, SimpleUpdate

makedocs(;
    modules=[SimpleUpdate],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/under-Peter/SimpleUpdate.jl/blob/{commit}{path}#L{line}",
    sitename="SimpleUpdate.jl",
    authors="Andreas Peter",
    assets=[],
)

deploydocs(;
    repo="github.com/under-Peter/SimpleUpdate.jl",
)
