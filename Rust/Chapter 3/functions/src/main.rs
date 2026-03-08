fn main() {
    println!("Hello, world!");

    another_function();

    let integer: i32 = 67;
    another_function_with_parameters(integer);

    print_labeled_measurements(10, 'h');

    let y: i32 = {
        let x = 3;
        x + 1
    };

    another_function_with_parameters(y);

    let integer: i32 = five();
    another_function_with_parameters(integer);

    let integer: i32 = five_with_return();
    another_function_with_parameters(integer);
}

fn another_function() {
    println!("Another function.");
}

fn another_function_with_parameters(x: i32) {
    println!("The value of x is: {x}");
}

fn print_labeled_measurements(value: i32, unit_label: char) {
    println!("The measurement is: {value}{unit_label}");
}

// Functions can return as expressions or with return statements

fn five() -> i32 {
    5
}

fn five_with_return() -> i32 {
    return 5;
}
