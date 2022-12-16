using TaylorDiff
using Documenter

DocMeta.setdocmeta!(TaylorDiff, :DocTestSetup, :(using TaylorDiff); recursive=true)

makedocs(;
    modules=[TaylorDiff],
    authors="Songchen Tan <i@tansongchen.com> and contributors",
    repo="https://github.com/tansongchen/TaylorDiff.jl/blob/{commit}{path}#{line}",
    sitename="TaylorDiff.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tansongchen.github.io/TaylorDiff.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tansongchen/TaylorDiff.jl",
    devbranch="main",
)
