using TaylorDiff
using Documenter

DocMeta.setdocmeta!(TaylorDiff, :DocTestSetup, :(using TaylorDiff); recursive = true)

makedocs(;
    modules = [TaylorDiff],
    authors = "Songchen Tan <i@tansongchen.com> and contributors",
    repo = "https://github.com/JuliaDiff/TaylorDiff.jl/blob/{commit}{path}#{line}",
    sitename = "TaylorDiff.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliadiff.org/TaylorDiff.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "Efficient Halley's method for nonlinear solving" => "examples/halley.md"
        ],
        "Theory" => "theory.md",
        "API" => "api.md"
    ])

deploydocs(;
    repo = "github.com/JuliaDiff/TaylorDiff.jl",
    devbranch = "main")
