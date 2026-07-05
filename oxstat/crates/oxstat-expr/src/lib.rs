//! oxstat-expr: Expression parser and evaluator.
//!
//! Handles COMPUTE, IF, RECODE expression parsing and evaluation.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use winnow::prelude::*;
use winnow::ascii::{float, space0, Caseless};
use winnow::combinator::{alt, fail, not, peek, terminated};
use winnow::token::{literal, one_of};

use oxstat_core::{Dataset, MissingValues, Value, Variable, VariableType};

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
    Pos,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Value(Value),
    Variable(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    FunctionCall {
        name: String,
        min_valid: Option<usize>,
        args: Vec<Expr>,
    },
}

// Case-insensitive keyword helper
fn keyword<'s>(kw: &'static str) -> impl FnMut(&mut &'s str) -> PResult<&'s str> {
    move |input: &mut &'s str| {
        terminated(
            literal(Caseless(kw)),
            peek(not(one_of(('a'..='z', 'A'..='Z', '0'..='9', '.', '_', '@', '#', '$'))))
        ).parse_next(input)
    }
}

// Single quoted string parsing
fn parse_single_quoted_string<'s>(input: &mut &'s str) -> PResult<String> {
    let _ = literal("'").parse_next(input)?;
    let mut s = String::new();
    loop {
        if let Some(rest) = input.strip_prefix("''") {
            *input = rest;
            s.push('\'');
        } else if let Some(rest) = input.strip_prefix('\'') {
            *input = rest;
            break;
        } else {
            if input.is_empty() {
                return fail.parse_next(input);
            }
            let next_char = input.chars().next().unwrap();
            *input = &input[next_char.len_utf8()...];
            s.push(next_char);
        }
    }
    Ok(s)
}

// Double quoted string parsing
fn parse_double_quoted_string<'s>(input: &mut &'s str) -> PResult<String> {
    let _ = literal("\"").parse_next(input)?;
    let mut s = String::new();
    loop {
        if let Some(rest) = input.strip_prefix("\"\"") {
            *input = rest;
            s.push('"');
        } else if let Some(rest) = input.strip_prefix('"') {
            *input = rest;
            break;
        } else {
            if input.is_empty() {
                return fail.parse_next(input);
            }
            let next_char = input.chars().next().unwrap();
            *input = &input[next_char.len_utf8()...];
            s.push(next_char);
        }
    }
    Ok(s)
}

// Identifier parsing
fn parse_identifier<'s>(input: &mut &'s str) -> PResult<String> {
    let first = one_of(('a'..='z', 'A'..='Z', '_', '$', '@', '#')).parse_next(input)?;
    let mut s = String::from(first);
    while let Some(c) = input.chars().next() {
        if c.is_alphanumeric() || c == '.' || c == '_' || c == '@' || c == '#' || c == '$' {
            *input = &input[c.len_utf8()...];
            s.push(c);
        } else {
            break;
        }
    }
    Ok(s)
}

fn parse_number<'s>(input: &mut &'s str) -> PResult<Value> {
    let val: f64 = float.parse_next(input)?;
    Ok(Value::Numeric(val))
}

fn parse_string<'s>(input: &mut &'s str) -> PResult<Value> {
    let s = alt((parse_single_quoted_string, parse_double_quoted_string)).parse_next(input)?;
    Ok(Value::String(s))
}

fn parse_function_call<'s>(input: &mut &'s str) -> PResult<Expr> {
    let checkpoint = *input;
    let name = parse_identifier.parse_next(input)?;
    let _ = space0.parse_next(input)?;
    if !input.starts_with('(') {
        *input = checkpoint;
        return fail.parse_next(&mut "");
    }
    let _ = literal("(").parse_next(input)?;
    let _ = space0.parse_next(input)?;

    let mut args = Vec::new();
    if !input.starts_with(')') {
        loop {
            let arg = parse_expr(input)?;
            args.push(arg);
            let _ = space0.parse_next(input)?;
            if let Some(rest) = input.strip_prefix(',') {
                *input = rest;
                let _ = space0.parse_next(input)?;
            } else {
                break;
            }
        }
    }

    let _ = literal(")").parse_next(input)?;

    let mut base_name = name;
    let mut min_valid = None;
    if let Some(dot_idx) = base_name.find('.') {
        let parts: Vec<&str> = base_name.split('.').collect();
        if parts.len() == 2 {
            if let Ok(num) = parts[1].parse::<usize>() {
                base_name = parts[0].to_string();
                min_valid = Some(num);
            }
        }
    }

    Ok(Expr::FunctionCall {
        name: base_name.to_uppercase(),
        min_valid,
        args,
    })
}

fn parse_variable<'s>(input: &mut &'s str) -> PResult<Expr> {
    let checkpoint = *input;
    let name = parse_identifier.parse_next(input)?;
    let upper = name.to_uppercase();
    if upper == "AND" || upper == "OR" || upper == "NOT" ||
       upper == "EQ" || upper == "NE" || upper == "LT" ||
       upper == "LE" || upper == "GT" || upper == "GE" {
        *input = checkpoint;
        return fail.parse_next(&mut "");
    }
    Ok(Expr::Variable(name))
}

fn parse_primary<'s>(input: &mut &'s str) -> PResult<Expr> {
    let _ = space0.parse_next(input)?;
    alt((
        parse_parenthesized,
        parse_function_call,
        parse_number.map(Expr::Value),
        parse_string.map(Expr::Value),
        parse_variable,
    )).parse_next(input)
}

fn parse_parenthesized<'s>(input: &mut &'s str) -> PResult<Expr> {
    let _ = literal("(").parse_next(input)?;
    let _ = space0.parse_next(input)?;
    let expr = parse_expr(input)?;
    let _ = space0.parse_next(input)?;
    let _ = literal(")").parse_next(input)?;
    Ok(expr)
}

fn parse_exponentiation<'s>(input: &mut &'s str) -> PResult<Expr> {
    let lhs = parse_primary.parse_next(input)?;
    let _ = space0.parse_next(input)?;
    if input.starts_with("**") {
        let _ = literal("**").parse_next(input)?;
        let _ = space0.parse_next(input)?;
        let rhs = parse_exponentiation(input)?;
        Ok(Expr::Binary(BinaryOp::Pow, Box::new(lhs), Box::new(rhs)))
    } else {
        Ok(lhs)
    }
}

fn parse_unary<'s>(input: &mut &'s str) -> PResult<Expr> {
    let _ = space0.parse_next(input)?;
    if let Some(rest) = input.strip_prefix('-') {
        *input = rest;
        let expr = parse_unary(input)?;
        Ok(Expr::Unary(UnaryOp::Neg, Box::new(expr)))
    } else if let Some(rest) = input.strip_prefix('+') {
        *input = rest;
        let expr = parse_unary(input)?;
        Ok(Expr::Unary(UnaryOp::Pos, Box::new(expr)))
    } else if let Some(rest) = input.strip_prefix('~') {
        *input = rest;
        let expr = parse_unary(input)?;
        Ok(Expr::Unary(UnaryOp::Not, Box::new(expr)))
    } else if let Some(rest) = input.strip_prefix('!') {
        *input = rest;
        let expr = parse_unary(input)?;
        Ok(Expr::Unary(UnaryOp::Not, Box::new(expr)))
    } else {
        let checkpoint = *input;
        if keyword("NOT").parse_next(input).is_ok() {
            let expr = parse_unary(input)?;
            Ok(Expr::Unary(UnaryOp::Not, Box::new(expr)))
        } else {
            *input = checkpoint;
            parse_exponentiation(input)
        }
    }
}

fn parse_multiplicative<'s>(input: &mut &'s str) -> PResult<Expr> {
    let mut lhs = parse_unary.parse_next(input)?;
    loop {
        let _ = space0.parse_next(input)?;
        if let Some(rest) = input.strip_prefix('*') {
            *input = rest;
            let rhs = parse_unary(input)?;
            lhs = Expr::Binary(BinaryOp::Mul, Box::new(lhs), Box::new(rhs));
        } else if let Some(rest) = input.strip_prefix('/') {
            *input = rest;
            let rhs = parse_unary(input)?;
            lhs = Expr::Binary(BinaryOp::Div, Box::new(lhs), Box::new(rhs));
        } else {
            break;
        }
    }
    Ok(lhs)
}

fn parse_additive<'s>(input: &mut &'s str) -> PResult<Expr> {
    let mut lhs = parse_multiplicative.parse_next(input)?;
    loop {
        let _ = space0.parse_next(input)?;
        if let Some(rest) = input.strip_prefix('+') {
            *input = rest;
            let rhs = parse_multiplicative(input)?;
            lhs = Expr::Binary(BinaryOp::Add, Box::new(lhs), Box::new(rhs));
        } else if let Some(rest) = input.strip_prefix('-') {
            *input = rest;
            let rhs = parse_multiplicative(input)?;
            lhs = Expr::Binary(BinaryOp::Sub, Box::new(lhs), Box::new(rhs));
        } else {
            break;
        }
    }
    Ok(lhs)
}

fn parse_relational<'s>(input: &mut &'s str) -> PResult<Expr> {
    let mut lhs = parse_additive.parse_next(input)?;
    loop {
        let _ = space0.parse_next(input)?;
        let checkpoint = *input;

        let op = if let Some(rest) = input.strip_prefix("<>") {
            *input = rest;
            Some(BinaryOp::Ne)
        } else if let Some(rest) = input.strip_prefix("~=") {
            *input = rest;
            Some(BinaryOp::Ne)
        } else if let Some(rest) = input.strip_prefix("!=") {
            *input = rest;
            Some(BinaryOp::Ne)
        } else if let Some(rest) = input.strip_prefix("<=") {
            *input = rest;
            Some(BinaryOp::Le)
        } else if let Some(rest) = input.strip_prefix(">=") {
            *input = rest;
            Some(BinaryOp::Ge)
        } else if let Some(rest) = input.strip_prefix('<') {
            *input = rest;
            Some(BinaryOp::Lt)
        } else if let Some(rest) = input.strip_prefix('>') {
            *input = rest;
            Some(BinaryOp::Gt)
        } else if let Some(rest) = input.strip_prefix('=') {
            *input = rest;
            Some(BinaryOp::Eq)
        } else if keyword("EQ").parse_next(input).is_ok() {
            Some(BinaryOp::Eq)
        } else if keyword("NE").parse_next(input).is_ok() {
            Some(BinaryOp::Ne)
        } else if keyword("LT").parse_next(input).is_ok() {
            Some(BinaryOp::Lt)
        } else if keyword("LE").parse_next(input).is_ok() {
            Some(BinaryOp::Le)
        } else if keyword("GT").parse_next(input).is_ok() {
            Some(BinaryOp::Gt)
        } else if keyword("GE").parse_next(input).is_ok() {
            Some(BinaryOp::Ge)
        } else {
            *input = checkpoint;
            None
        };

        if let Some(bin_op) = op {
            let rhs = parse_additive(input)?;
            lhs = Expr::Binary(bin_op, Box::new(lhs), Box::new(rhs));
        } else {
            break;
        }
    }
    Ok(lhs)
}

fn parse_and<'s>(input: &mut &'s str) -> PResult<Expr> {
    let mut lhs = parse_relational.parse_next(input)?;
    loop {
        let _ = space0.parse_next(input)?;
        let checkpoint = *input;
        let is_and = if let Some(rest) = input.strip_prefix('&') {
            *input = rest;
            true
        } else if keyword("AND").parse_next(input).is_ok() {
            true
        } else {
            *input = checkpoint;
            false
        };

        if is_and {
            let rhs = parse_relational(input)?;
            lhs = Expr::Binary(BinaryOp::And, Box::new(lhs), Box::new(rhs));
        } else {
            break;
        }
    }
    Ok(lhs)
}

fn parse_or<'s>(input: &mut &'s str) -> PResult<Expr> {
    let mut lhs = parse_and.parse_next(input)?;
    loop {
        let _ = space0.parse_next(input)?;
        let checkpoint = *input;
        let is_or = if let Some(rest) = input.strip_prefix('|') {
            *input = rest;
            true
        } else if keyword("OR").parse_next(input).is_ok() {
            true
        } else {
            *input = checkpoint;
            false
        };

        if is_or {
            let rhs = parse_and(input)?;
            lhs = Expr::Binary(BinaryOp::Or, Box::new(lhs), Box::new(rhs));
        } else {
            break;
        }
    }
    Ok(lhs)
}

fn parse_expr<'s>(input: &mut &'s str) -> PResult<Expr> {
    parse_or(input)
}

pub fn parse(input: &str) -> Result<Expr, String> {
    let mut input_ref = input.trim();
    let expr = parse_expr(&mut input_ref)
        .map_err(|e| format!("Parsing error: {:?}", e))?;
    if !input_ref.is_empty() {
        return Err(format!("Trailing characters: '{}'", input_ref));
    }
    Ok(expr)
}

// Date helpers
fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_in_month(month: u32, year: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap_year(year) { 29 } else { 28 },
        _ => 0,
    }
}

fn days_since_1582_10_14(day: u32, month: u32, year: u32) -> Option<f64> {
    if year < 1582 || month < 1 || month > 12 || day < 1 {
        return None;
    }
    if year == 1582 && (month < 10 || (month == 10 && day < 14)) {
        return None;
    }
    if day > days_in_month(month, year) {
        return None;
    }

    let mut days = 0.0;
    for y in 1582..year {
        if y == 1582 {
            days += 78.0;
        } else {
            days += if is_leap_year(y) { 366.0 } else { 365.0 };
        }
    }

    if year == 1582 {
        if month == 10 {
            days += (day - 14) as f64;
        } else if month == 11 {
            days += 17.0 + (day - 1) as f64;
        } else if month == 12 {
            days += 17.0 + 30.0 + (day - 1) as f64;
        }
    } else {
        for m in 1..month {
            days += days_in_month(m, year) as f64;
        }
        days += (day - 1) as f64;
    }

    Some(days)
}

fn seconds_to_date(seconds: f64) -> Option<(u32, u32, u32)> {
    if seconds < 0.0 {
        return None;
    }
    let mut days = (seconds / 86400.0).floor();

    if days < 78.0 {
        let mut d = days;
        if d < 17.0 {
            return Some((14 + d as u32, 10, 1582));
        }
        d -= 17.0;
        if d < 30.0 {
            return Some((1 + d as u32, 11, 1582));
        }
        d -= 30.0;
        return Some((1 + d as u32, 12, 1582));
    }

    days -= 78.0;
    let mut year = 1583;
    loop {
        let yr_days = if is_leap_year(year) { 366.0 } else { 365.0 };
        if days < yr_days {
            break;
        }
        days -= yr_days;
        year += 1;
    }

    let mut month = 1;
    loop {
        let m_days = days_in_month(month, year) as f64;
        if days < m_days {
            break;
        }
        days -= m_days;
        month += 1;
    }

    Some((1 + days as u32, month, year))
}

fn get_variable_value(
    dataset: &Dataset,
    batch_idx: usize,
    row_idx: usize,
    name: &str,
    ignore_user_missing: bool,
) -> Value {
    let var_idx = match dataset.variable_index(name) {
        Some(idx) => idx,
        None => return Value::SystemMissing,
    };
    let var = &dataset.variables[var_idx];
    let batch = &dataset.batches[batch_idx];
    let column = batch.column(var_idx);

    match var.var_type {
        VariableType::Numeric => {
            if column.is_null(row_idx) {
                Value::SystemMissing
            } else {
                let array = column
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                let val = array.value(row_idx);
                if !ignore_user_missing && var.missing.is_user_missing(val) {
                    Value::SystemMissing
                } else {
                    Value::Numeric(val)
                }
            }
        }
        VariableType::String(_) => {
            if column.is_null(row_idx) {
                Value::SystemMissing
            } else {
                let array = column
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .unwrap();
                let val = array.value(row_idx);
                Value::String(val.to_string())
            }
        }
    }
}

fn eval_aggregate(
    name: &str,
    min_valid: Option<usize>,
    args: &[Expr],
    dataset: &Dataset,
    batch_idx: usize,
    row_idx: usize,
) -> Value {
    let valid_nums: Vec<f64> = args.iter()
        .map(|arg| arg.eval(dataset, batch_idx, row_idx))
        .filter_map(|v| match v {
            Value::Numeric(n) => Some(n),
            _ => None,
        })
        .collect();

    let required = min_valid.unwrap_or(match name {
        "SD" | "VARIANCE" => 2,
        _ => 1,
    });

    if valid_nums.len() < required {
        return Value::SystemMissing;
    }

    match name {
        "SUM" => Value::Numeric(valid_nums.iter().sum()),
        "MEAN" => {
            let sum: f64 = valid_nums.iter().sum();
            Value::Numeric(sum / valid_nums.len() as f64)
        }
        "MIN" => {
            let min_val = valid_nums.iter().copied().fold(f64::INFINITY, f64::min);
            Value::Numeric(min_val)
        }
        "MAX" => {
            let max_val = valid_nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Value::Numeric(max_val)
        }
        "VARIANCE" => {
            let n = valid_nums.len() as f64;
            let mean: f64 = valid_nums.iter().sum::<f64>() / n;
            let sq_sum: f64 = valid_nums.iter().map(|&x| (x - mean).powi(2)).sum();
            Value::Numeric(sq_sum / (n - 1.0))
        }
        "SD" => {
            let n = valid_nums.len() as f64;
            let mean: f64 = valid_nums.iter().sum::<f64>() / n;
            let sq_sum: f64 = valid_nums.iter().map(|&x| (x - mean).powi(2)).sum();
            let var = sq_sum / (n - 1.0);
            Value::Numeric(var.sqrt())
        }
        _ => Value::SystemMissing,
    }
}

fn eval_function_call(
    name: &str,
    min_valid: Option<usize>,
    args: &[Expr],
    dataset: &Dataset,
    batch_idx: usize,
    row_idx: usize,
) -> Value {
    if matches!(name, "SUM" | "MEAN" | "MIN" | "MAX" | "VARIANCE" | "SD") {
        return eval_aggregate(name, min_valid, args, dataset, batch_idx, row_idx);
    }

    if name == "VALUE" {
        if args.len() == 1 {
            if let Expr::Variable(vname) = &args[0] {
                return get_variable_value(dataset, batch_idx, row_idx, vname, true);
            }
        }
        return Value::SystemMissing;
    }
    if name == "MISSING" {
        if args.len() == 1 {
            let val = args[0].eval(dataset, batch_idx, row_idx);
            return Value::Numeric(if val == Value::SystemMissing { 1.0 } else { 0.0 });
        }
        return Value::SystemMissing;
    }
    if name == "SYSMIS" {
        if args.len() == 1 {
            if let Expr::Variable(vname) = &args[0] {
                let raw_val = get_variable_value(dataset, batch_idx, row_idx, vname, true);
                return Value::Numeric(if raw_val == Value::SystemMissing { 1.0 } else { 0.0 });
            } else {
                let val = args[0].eval(dataset, batch_idx, row_idx);
                return Value::Numeric(if val == Value::SystemMissing { 1.0 } else { 0.0 });
            }
        }
        return Value::SystemMissing;
    }
    if name == "NMISS" {
        let mut missing_count = 0;
        for arg in args {
            if arg.eval(dataset, batch_idx, row_idx) == Value::SystemMissing {
                missing_count += 1;
            }
        }
        return Value::Numeric(missing_count as f64);
    }
    if name == "NVALID" {
        let mut valid_count = 0;
        for arg in args {
            if arg.eval(dataset, batch_idx, row_idx) != Value::SystemMissing {
                valid_count += 1;
            }
        }
        return Value::Numeric(valid_count as f64);
    }

    let evaluated_args: Vec<Value> = args.iter()
        .map(|arg| arg.eval(dataset, batch_idx, row_idx))
        .collect();

    match name {
        "ABS" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    return Value::Numeric(x.abs());
                }
            }
            Value::SystemMissing
        }
        "SQRT" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    if x >= 0.0 {
                        return Value::Numeric(x.sqrt());
                    }
                }
            }
            Value::SystemMissing
        }
        "EXP" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    return Value::Numeric(x.exp());
                }
            }
            Value::SystemMissing
        }
        "LN" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    if x > 0.0 {
                        return Value::Numeric(x.ln());
                    }
                }
            }
            Value::SystemMissing
        }
        "LG10" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    if x > 0.0 {
                        return Value::Numeric(x.log10());
                    }
                }
            }
            Value::SystemMissing
        }
        "RND" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    return Value::Numeric(x.round());
                }
            } else if evaluated_args.len() == 2 {
                if let (Value::Numeric(x), Value::Numeric(mult)) = (&evaluated_args[0], &evaluated_args[1]) {
                    if *mult != 0.0 {
                        return Value::Numeric((x / mult).round() * mult);
                    }
                }
            }
            Value::SystemMissing
        }
        "TRUNC" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    return Value::Numeric(x.trunc());
                }
            } else if evaluated_args.len() == 2 {
                if let (Value::Numeric(x), Value::Numeric(mult)) = (&evaluated_args[0], &evaluated_args[1]) {
                    if *mult != 0.0 {
                        return Value::Numeric((x / mult).trunc() * mult);
                    }
                }
            }
            Value::SystemMissing
        }
        "MOD" => {
            if evaluated_args.len() == 2 {
                if let (Value::Numeric(x), Value::Numeric(y)) = (&evaluated_args[0], &evaluated_args[1]) {
                    if *y != 0.0 {
                        return Value::Numeric(x % y);
                    }
                }
            }
            Value::SystemMissing
        }
        "SIN" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    return Value::Numeric(x.sin());
                }
            }
            Value::SystemMissing
        }
        "COS" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    return Value::Numeric(x.cos());
                }
            }
            Value::SystemMissing
        }
        "TAN" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    return Value::Numeric(x.tan());
                }
            }
            Value::SystemMissing
        }
        "ARSIN" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    if x >= -1.0 && x <= 1.0 {
                        return Value::Numeric(x.asin());
                    }
                }
            }
            Value::SystemMissing
        }
        "ARCOS" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    if x >= -1.0 && x <= 1.0 {
                        return Value::Numeric(x.acos());
                    }
                }
            }
            Value::SystemMissing
        }
        "ARTAN" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(x) = evaluated_args[0] {
                    return Value::Numeric(x.atan());
                }
            }
            Value::SystemMissing
        }

        "CONCAT" => {
            let mut s = String::new();
            for val in evaluated_args {
                match val {
                    Value::String(val_str) => s.push_str(&val_str),
                    _ => return Value::SystemMissing,
                }
            }
            Value::String(s)
        }
        "LOWER" | "LOWERCASE" => {
            if evaluated_args.len() == 1 {
                if let Value::String(s) = &evaluated_args[0] {
                    return Value::String(s.to_lowercase());
                }
            }
            Value::SystemMissing
        }
        "UPPER" | "UPPERCASE" => {
            if evaluated_args.len() == 1 {
                if let Value::String(s) = &evaluated_args[0] {
                    return Value::String(s.to_uppercase());
                }
            }
            Value::SystemMissing
        }
        "LENGTH" | "CHAR.LENGTH" => {
            if evaluated_args.len() == 1 {
                if let Value::String(s) = &evaluated_args[0] {
                    return Value::Numeric(s.len() as f64);
                }
            }
            Value::SystemMissing
        }
        "SUBSTR" | "CHAR.SUBSTR" => {
            if evaluated_args.len() == 2 {
                if let (Value::String(s), Value::Numeric(pos)) = (&evaluated_args[0], &evaluated_args[1]) {
                    let pos_idx = *pos as isize;
                    if pos_idx <= 0 || pos_idx as usize > s.len() {
                        return Value::String(String::new());
                    }
                    let start = (pos_idx - 1) as usize;
                    return Value::String(s[start..].to_string());
                }
            } else if evaluated_args.len() == 3 {
                if let (Value::String(s), Value::Numeric(pos), Value::Numeric(len)) = (&evaluated_args[0], &evaluated_args[1], &evaluated_args[2]) {
                    let pos_idx = *pos as isize;
                    let length = *len as usize;
                    if pos_idx <= 0 || pos_idx as usize > s.len() {
                        return Value::String(String::new());
                    }
                    let start = (pos_idx - 1) as usize;
                    let end = (start + length).min(s.len());
                    return Value::String(s[start..end].to_string());
                }
            }
            Value::SystemMissing
        }
        "LPAD" => {
            if evaluated_args.len() == 2 {
                if let (Value::String(s), Value::Numeric(len)) = (&evaluated_args[0], &evaluated_args[1]) {
                    let length = *len as usize;
                    if s.len() >= length {
                        return Value::String(s.clone());
                    }
                    let pad_len = length - s.len();
                    let pad = " ".repeat(pad_len);
                    return Value::String(format!("{}{}", pad, s));
                }
            } else if evaluated_args.len() == 3 {
                if let (Value::String(s), Value::Numeric(len), Value::String(pad)) = (&evaluated_args[0], &evaluated_args[1], &evaluated_args[2]) {
                    let length = *len as usize;
                    if s.len() >= length || pad.is_empty() {
                        return Value::String(s.clone());
                    }
                    let pad_char = pad.chars().next().unwrap();
                    let pad_len = length - s.len();
                    let pad_str = std::iter::repeat(pad_char).take(pad_len).collect::<String>();
                    return Value::String(format!("{}{}", pad_str, s));
                }
            }
            Value::SystemMissing
        }
        "RPAD" => {
            if evaluated_args.len() == 2 {
                if let (Value::String(s), Value::Numeric(len)) = (&evaluated_args[0], &evaluated_args[1]) {
                    let length = *len as usize;
                    if s.len() >= length {
                        return Value::String(s.clone());
                    }
                    let pad_len = length - s.len();
                    let pad = " ".repeat(pad_len);
                    return Value::String(format!("{}{}", s, pad));
                }
            } else if evaluated_args.len() == 3 {
                if let (Value::String(s), Value::Numeric(len), Value::String(pad)) = (&evaluated_args[0], &evaluated_args[1], &evaluated_args[2]) {
                    let length = *len as usize;
                    if s.len() >= length || pad.is_empty() {
                        return Value::String(s.clone());
                    }
                    let pad_char = pad.chars().next().unwrap();
                    let pad_len = length - s.len();
                    let pad_str = std::iter::repeat(pad_char).take(pad_len).collect::<String>();
                    return Value::String(format!("{}{}", s, pad_str));
                }
            }
            Value::SystemMissing
        }
        "LTRIM" => {
            if evaluated_args.len() == 1 {
                if let Value::String(s) = &evaluated_args[0] {
                    return Value::String(s.trim_start().to_string());
                }
            } else if evaluated_args.len() == 2 {
                if let (Value::String(s), Value::String(c)) = (&evaluated_args[0], &evaluated_args[1]) {
                    if let Some(trim_char) = c.chars().next() {
                        return Value::String(s.trim_start_matches(trim_char).to_string());
                    }
                    return Value::String(s.clone());
                }
            }
            Value::SystemMissing
        }
        "RTRIM" => {
            if evaluated_args.len() == 1 {
                if let Value::String(s) = &evaluated_args[0] {
                    return Value::String(s.trim_end().to_string());
                }
            } else if evaluated_args.len() == 2 {
                if let (Value::String(s), Value::String(c)) = (&evaluated_args[0], &evaluated_args[1]) {
                    if let Some(trim_char) = c.chars().next() {
                        return Value::String(s.trim_end_matches(trim_char).to_string());
                    }
                    return Value::String(s.clone());
                }
            }
            Value::SystemMissing
        }
        "INDEX" | "CHAR.INDEX" => {
            if evaluated_args.len() == 2 {
                if let (Value::String(s), Value::String(needle)) = (&evaluated_args[0], &evaluated_args[1]) {
                    if let Some(idx) = s.find(needle) {
                        return Value::Numeric((idx + 1) as f64);
                    } else {
                        return Value::Numeric(0.0);
                    }
                }
            }
            Value::SystemMissing
        }

        // Date Functions
        "DATE.DMY" => {
            if evaluated_args.len() == 3 {
                if let (Value::Numeric(d), Value::Numeric(m), Value::Numeric(y)) = (&evaluated_args[0], &evaluated_args[1], &evaluated_args[2]) {
                    if let Some(days) = days_since_1582_10_14(*d as u32, *m as u32, *y as u32) {
                        return Value::Numeric(days * 86400.0);
                    }
                }
            }
            Value::SystemMissing
        }
        "DATE.MDY" => {
            if evaluated_args.len() == 3 {
                if let (Value::Numeric(m), Value::Numeric(d), Value::Numeric(y)) = (&evaluated_args[0], &evaluated_args[1], &evaluated_args[2]) {
                    if let Some(days) = days_since_1582_10_14(*d as u32, *m as u32, *y as u32) {
                        return Value::Numeric(days * 86400.0);
                    }
                }
            }
            Value::SystemMissing
        }
        "TIME.HMS" => {
            if evaluated_args.len() == 3 {
                if let (Value::Numeric(h), Value::Numeric(m), Value::Numeric(s)) = (&evaluated_args[0], &evaluated_args[1], &evaluated_args[2]) {
                    return Value::Numeric(*h * 3600.0 + *m * 60.0 + *s);
                }
            }
            Value::SystemMissing
        }
        "TIME.DAYS" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(d) = evaluated_args[0] {
                    return Value::Numeric(d * 86400.0);
                }
            }
            Value::SystemMissing
        }
        "XDATE.YEAR" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    if let Some((_, _, year)) = seconds_to_date(t) {
                        return Value::Numeric(year as f64);
                    }
                }
            }
            Value::SystemMissing
        }
        "XDATE.MONTH" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    if let Some((_, month, _)) = seconds_to_date(t) {
                        return Value::Numeric(month as f64);
                    }
                }
            }
            Value::SystemMissing
        }
        "XDATE.MDAY" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    if let Some((day, _, _)) = seconds_to_date(t) {
                        return Value::Numeric(day as f64);
                    }
                }
            }
            Value::SystemMissing
        }
        "XDATE.WKDAY" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    let days = (t / 86400.0).floor();
                    if days >= 0.0 {
                        let wkday = ((days as i64 + 4) % 7) + 1;
                        return Value::Numeric(wkday as f64);
                    }
                }
            }
            Value::SystemMissing
        }
        "XDATE.HOUR" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    if t >= 0.0 {
                        let hour = ((t / 3600.0).floor() as i64) % 24;
                        return Value::Numeric(hour as f64);
                    }
                }
            }
            Value::SystemMissing
        }
        "XDATE.MINUTE" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    if t >= 0.0 {
                        let minute = ((t / 60.0).floor() as i64) % 60;
                        return Value::Numeric(minute as f64);
                    }
                }
            }
            Value::SystemMissing
        }
        "XDATE.SECOND" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    if t >= 0.0 {
                        let second = (t.floor() as i64) % 60;
                        return Value::Numeric(second as f64);
                    }
                }
            }
            Value::SystemMissing
        }
        "CTIME.DAYS" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    return Value::Numeric(t / 86400.0);
                }
            }
            Value::SystemMissing
        }
        "CTIME.HOURS" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    return Value::Numeric(t / 3600.0);
                }
            }
            Value::SystemMissing
        }
        "CTIME.MINUTES" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    return Value::Numeric(t / 60.0);
                }
            }
            Value::SystemMissing
        }
        "CTIME.SECONDS" => {
            if evaluated_args.len() == 1 {
                if let Value::Numeric(t) = evaluated_args[0] {
                    return Value::Numeric(t);
                }
            }
            Value::SystemMissing
        }

        _ => Value::SystemMissing,
    }
}

impl Expr {
    pub fn eval(&self, dataset: &Dataset, batch_idx: usize, row_idx: usize) -> Value {
        match self {
            Expr::Value(val) => val.clone(),
            Expr::Variable(name) => {
                get_variable_value(dataset, batch_idx, row_idx, name, false)
            }
            Expr::Unary(op, expr) => {
                let val = expr.eval(dataset, batch_idx, row_idx);
                match op {
                    UnaryOp::Neg => match val {
                        Value::Numeric(v) => Value::Numeric(-v),
                        _ => Value::SystemMissing,
                    },
                    UnaryOp::Pos => match val {
                        Value::Numeric(v) => Value::Numeric(v),
                        _ => Value::SystemMissing,
                    },
                    UnaryOp::Not => match val {
                        Value::Numeric(v) => {
                            if v == 0.0 {
                                Value::Numeric(1.0)
                            } else if v == 1.0 {
                                Value::Numeric(0.0)
                            } else {
                                Value::SystemMissing
                            }
                        }
                        Value::SystemMissing => Value::SystemMissing,
                        _ => Value::SystemMissing,
                    },
                }
            }
            Expr::Binary(op, lhs, rhs) => {
                let left = lhs.eval(dataset, batch_idx, row_idx);

                if *op == BinaryOp::And {
                    match left {
                        Value::Numeric(lv) if lv == 0.0 => return Value::Numeric(0.0),
                        _ => {}
                    }
                    let right = rhs.eval(dataset, batch_idx, row_idx);
                    return match (left, right) {
                        (Value::Numeric(lv), Value::Numeric(rv)) => {
                            if lv != 0.0 && rv != 0.0 {
                                Value::Numeric(1.0)
                            } else if lv == 0.0 || rv == 0.0 {
                                Value::Numeric(0.0)
                            } else {
                                Value::SystemMissing
                            }
                        }
                        (Value::Numeric(lv), Value::SystemMissing) => {
                            if lv == 0.0 {
                                Value::Numeric(0.0)
                            } else {
                                Value::SystemMissing
                            }
                        }
                        (Value::SystemMissing, Value::Numeric(rv)) => {
                            if rv == 0.0 {
                                Value::Numeric(0.0)
                            } else {
                                Value::SystemMissing
                            }
                        }
                        _ => Value::SystemMissing,
                    };
                }

                if *op == BinaryOp::Or {
                    match left {
                        Value::Numeric(lv) if lv != 0.0 => return Value::Numeric(1.0),
                        _ => {}
                    }
                    let right = rhs.eval(dataset, batch_idx, row_idx);
                    return match (left, right) {
                        (Value::Numeric(lv), Value::Numeric(rv)) => {
                            if lv != 0.0 || rv != 0.0 {
                                Value::Numeric(1.0)
                            } else {
                                Value::Numeric(0.0)
                            }
                        }
                        (Value::Numeric(lv), Value::SystemMissing) => {
                            if lv != 0.0 {
                                Value::Numeric(1.0)
                            } else {
                                Value::SystemMissing
                            }
                        }
                        (Value::SystemMissing, Value::Numeric(rv)) => {
                            if rv != 0.0 {
                                Value::Numeric(1.0)
                            } else {
                                Value::SystemMissing
                            }
                        }
                        _ => Value::SystemMissing,
                    };
                }

                let right = rhs.eval(dataset, batch_idx, row_idx);
                match (left, right) {
                    (Value::Numeric(l), Value::Numeric(r)) => match op {
                        BinaryOp::Add => Value::Numeric(l + r),
                        BinaryOp::Sub => Value::Numeric(l - r),
                        BinaryOp::Mul => Value::Numeric(l * r),
                        BinaryOp::Div => {
                            if r == 0.0 {
                                Value::SystemMissing
                            } else {
                                Value::Numeric(l / r)
                            }
                        }
                        BinaryOp::Pow => {
                            let res = l.powf(r);
                            if res.is_nan() || res.is_infinite() {
                                Value::SystemMissing
                            } else {
                                Value::Numeric(res)
                            }
                        }
                        BinaryOp::Eq => Value::Numeric(if l == r { 1.0 } else { 0.0 }),
                        BinaryOp::Ne => Value::Numeric(if l != r { 1.0 } else { 0.0 }),
                        BinaryOp::Lt => Value::Numeric(if l < r { 1.0 } else { 0.0 }),
                        BinaryOp::Le => Value::Numeric(if l <= r { 1.0 } else { 0.0 }),
                        BinaryOp::Gt => Value::Numeric(if l > r { 1.0 } else { 0.0 }),
                        BinaryOp::Ge => Value::Numeric(if l >= r { 1.0 } else { 0.0 }),
                        _ => Value::SystemMissing,
                    },
                    (Value::String(l), Value::String(r)) => match op {
                        BinaryOp::Eq => Value::Numeric(if l == r { 1.0 } else { 0.0 }),
                        BinaryOp::Ne => Value::Numeric(if l != r { 1.0 } else { 0.0 }),
                        BinaryOp::Lt => Value::Numeric(if l < r { 1.0 } else { 0.0 }),
                        BinaryOp::Le => Value::Numeric(if l <= r { 1.0 } else { 0.0 }),
                        BinaryOp::Gt => Value::Numeric(if l > r { 1.0 } else { 0.0 }),
                        BinaryOp::Ge => Value::Numeric(if l >= r { 1.0 } else { 0.0 }),
                        _ => Value::SystemMissing,
                    },
                    _ => Value::SystemMissing,
                }
            }
            Expr::FunctionCall { name, min_valid, args } => {
                eval_function_call(name, *min_valid, args, dataset, batch_idx, row_idx)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn create_mock_dataset() -> Dataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("X", DataType::Float64, true),
            Field::new("Y", DataType::Float64, true),
            Field::new("S", DataType::Utf8, true),
        ]));

        let x_arr = Arc::new(Float64Array::from(vec![Some(10.0), Some(20.0), None, Some(99.0)])) as Arc<dyn Array>;
        let y_arr = Arc::new(Float64Array::from(vec![Some(2.0), Some(0.0), Some(5.0), Some(4.0)])) as Arc<dyn Array>;
        let s_arr = Arc::new(StringArray::from(vec![Some("apple"), Some("banana"), None, Some("pear")])) as Arc<dyn Array>;

        let batch = RecordBatch::try_new(schema, vec![x_arr, y_arr, s_arr]).unwrap();

        let mut dataset = Dataset::new();
        dataset.batches.push(batch);

        // Add variables with metadata
        let mut x_var = Variable::numeric("X");
        // Let's set 99.0 as a user-missing value
        x_var.missing.discrete.push(99.0);
        dataset.add_variable(x_var);

        dataset.add_variable(Variable::numeric("Y"));
        dataset.add_variable(Variable::string("S", 10));

        dataset
    }

    #[test]
    fn test_parser_basic() {
        let parsed = parse("1 + 2 * 3").unwrap();
        assert_eq!(
            parsed,
            Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::Value(Value::Numeric(1.0))),
                Box::new(Expr::Binary(
                    BinaryOp::Mul,
                    Box::new(Expr::Value(Value::Numeric(2.0))),
                    Box::new(Expr::Value(Value::Numeric(3.0)))
                ))
            )
        );

        let parsed_pow = parse("2 ** 3 ** 2").unwrap();
        assert_eq!(
            parsed_pow,
            Expr::Binary(
                BinaryOp::Pow,
                Box::new(Expr::Value(Value::Numeric(2.0))),
                Box::new(Expr::Binary(
                    BinaryOp::Pow,
                    Box::new(Expr::Value(Value::Numeric(3.0))),
                    Box::new(Expr::Value(Value::Numeric(2.0)))
                ))
            )
        );

        let parsed_neg = parse("-X").unwrap();
        assert_eq!(
            parsed_neg,
            Expr::Unary(UnaryOp::Neg, Box::new(Expr::Variable("X".to_string())))
        );
    }

    #[test]
    fn test_parser_logical() {
        let parsed = parse("X > 5 AND Y < 10").unwrap();
        assert_eq!(
            parsed,
            Expr::Binary(
                BinaryOp::And,
                Box::new(Expr::Binary(
                    BinaryOp::Gt,
                    Box::new(Expr::Variable("X".to_string())),
                    Box::new(Expr::Value(Value::Numeric(5.0)))
                )),
                Box::new(Expr::Binary(
                    BinaryOp::Lt,
                    Box::new(Expr::Variable("Y".to_string())),
                    Box::new(Expr::Value(Value::Numeric(10.0)))
                ))
            )
        );
    }

    #[test]
    fn test_eval_basic() {
        let dataset = create_mock_dataset();

        // Row 0: X = 10, Y = 2
        let expr = parse("X + Y").unwrap();
        assert_eq!(expr.eval(&dataset, 0, 0), Value::Numeric(12.0));

        // Division by zero -> SystemMissing
        let div_expr = parse("X / Y").unwrap();
        assert_eq!(div_expr.eval(&dataset, 0, 0), Value::Numeric(5.0));
        assert_eq!(div_expr.eval(&dataset, 0, 1), Value::SystemMissing); // Y is 0.0

        // String operations
        let eq_expr = parse("S = 'apple'").unwrap();
        assert_eq!(eq_expr.eval(&dataset, 0, 0), Value::Numeric(1.0));
        assert_eq!(eq_expr.eval(&dataset, 0, 1), Value::Numeric(0.0));
    }

    #[test]
    fn test_eval_missing_values() {
        let dataset = create_mock_dataset();

        // Row 2: X is SYSMIS
        let expr = parse("X + Y").unwrap();
        assert_eq!(expr.eval(&dataset, 0, 2), Value::SystemMissing);

        // Row 3: X is 99.0, which is user-defined missing
        assert_eq!(expr.eval(&dataset, 0, 3), Value::SystemMissing);

        // VALUE(X) ignores user-defined missing -> 99.0 + 4.0 = 103.0
        let value_expr = parse("VALUE(X) + Y").unwrap();
        assert_eq!(value_expr.eval(&dataset, 0, 3), Value::Numeric(103.0));

        // MISSING(X) and SYSMIS(X)
        let missing_expr = parse("MISSING(X)").unwrap();
        let sysmis_expr = parse("SYSMIS(X)").unwrap();
        assert_eq!(missing_expr.eval(&dataset, 0, 3), Value::Numeric(1.0)); // 99.0 is missing
        assert_eq!(sysmis_expr.eval(&dataset, 0, 3), Value::Numeric(0.0));  // but not system-missing
    }

    #[test]
    fn test_eval_date_functions() {
        let dataset = create_mock_dataset();

        // DATE.DMY(14, 10, 1582) = 0.0
        let date_expr = parse("DATE.DMY(14, 10, 1582)").unwrap();
        assert_eq!(date_expr.eval(&dataset, 0, 0), Value::Numeric(0.0));

        // XDATE.YEAR(DATE.DMY(15, 6, 2026)) = 2026
        let xdate_expr = parse("XDATE.YEAR(DATE.DMY(15, 6, 2026))").unwrap();
        assert_eq!(xdate_expr.eval(&dataset, 0, 0), Value::Numeric(2026.0));
    }
}

