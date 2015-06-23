// precedence and associativity taken from
// http://sweetjs.org/doc/main/sweet.html#custom-operators
operator +   14 { $r } => #{ ad_add(0, $r) }
operator -   14 { $r } => #{ ad_sub(0, $r) }
operator *   13 left { $l, $r } => #{ ad_mul($l, $r) }
operator /   13 left { $l, $r } => #{ ad_div($l, $r) }
operator +   12 left { $l, $r } => #{ ad_add($l, $r) }
operator -   12 left { $l, $r } => #{ ad_sub($l, $r) }
operator <   10 left { $l, $r } => #{ ad_lt($l, $r) }
operator <=  10 left { $l, $r } => #{ ad_leq($l, $r) }
operator >   10 left { $l, $r } => #{ ad_gt($l, $r) }
operator >=  10 left { $l, $r } => #{ ad_geq($l, $r) }
operator ==   9 left { $l, $r } => #{ ad_eq($l, $r) }
operator !=   9 left { $l, $r } => #{ ad_neq($l, $r) }
operator ===  9 left { $l, $r } => #{ ad_peq($l, $r) }
operator !==  9 left { $l, $r } => #{ ad_pneq($l, $r) }
// needswork: modulo operation
// needswork: binary and bitwise operations


// Replace references to Math.x with ad_Math.x
macro Math {
	rule { .$x } => { ad_Math.$x }
}