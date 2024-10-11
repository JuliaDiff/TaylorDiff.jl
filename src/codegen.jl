for unary_func in (
    deg2rad, rad2deg,
    sinh, cosh, tanh,
    asin, acos, atan, asec, acsc, acot,
    log, log10, log1p, log2,
    asinh, acosh, atanh, asech, acsch,
    acoth,
    abs, sign)
    define_unary_function(unary_func, TaylorDiff)
end
