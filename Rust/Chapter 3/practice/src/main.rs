fn main() {
    println!("Solving Practice Problems");

    let fahrenheit: f32 = 60.0;
    let celsius: f32 = fahrenheit_to_celsius(fahrenheit);

    println!("{fahrenheit} in fahrenheit is: {celsius} celsius");

    let celsius: f32 = 20.0;
    let fahrenheit: f32 = celsius_to_fahrenheit(celsius);

    println!("{celsius} in celsius is: {fahrenheit} fahrenheit");

    let fib: i32 = 7;
    let nth_fib: i32 = nth_fibonacci(fib);

    println!("The {fib} fibonacci number is: {nth_fib}");
}

// (x * (9/5)) + 32
fn celsius_to_fahrenheit(x: f32) -> f32 {
    (x * (9.0/5.0)) + 32.0
}

// (x - 32) * (5/9)
fn fahrenheit_to_celsius(x: f32) -> f32 {
    (x - 32.0) * (5.0/9.0)
}

fn nth_fibonacci(x: i32) -> i32 {
    if x < 0 {
        return 0;
    }

    if x == 0 || x == 1 {
        return x;
    }

    nth_fibonacci(x - 1) + nth_fibonacci(x - 2)
}
