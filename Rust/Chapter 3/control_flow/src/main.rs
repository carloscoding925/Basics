fn main() {
    println!("Hello, world!");

    branches();

    let condition: bool = true;
    let number: i32 = if condition { 5 } else { 6 };
    println!("The value of number is: {number}");

    loops();
}

// We can return the unit type to show that we're not
// actually returning anything from this function
fn branches() -> () {
    let number: i32 = 3;

    // The condition in an if statement must be a bool
    if number < 5 {
        println!("condition was true");
    }
    else {
        println!("condition was false");
    }

    if number != 0 {
        println!("Number was something other than zero");
    }

    if number % 4 == 0 {
        println!("Number is divisible by 4");
    }
    else if number % 3 == 0 {
        println!("Number is divisible by 3");
    }
    else {
        println!("Number is not divisible by 4 or 3");
    }
}

fn loops() -> () {
    let mut count: i32 = 0;
    loop {
        println!("again!");

        count += 1;
        if count > 5 {
            break;
        }
    }

    let mut counter: i32 = 0;
    let result: i32 = loop {
        counter += 1;

        if counter == 10 {
            break counter * 2;
        }
    };

    println!("The result is: {result}");

    let mut count: i32 = 0;
    'counting_up: loop {
        println!("count = {count}");
        let mut remaining: i32 = 10;

        loop {
            println!("remaining = {remaining}");
            if remaining == 9 {
                break;
            }
            if count == 2 {
                break 'counting_up;
            }
            remaining -= 1;
        }

        count += 1;
    }
    println!("End count = {count}");

    let mut number: i32 = 3;

    while number != 0 {
        println!("{number}!");

        number -= 1;
    }
    println!("Liftoff!");

    let a: [i32; 5] = [10, 20, 30, 40, 50];
    let mut index = 0;

    while index < 5 {
        println!("the value is: {}", a[index]);

        index += 1;
    }

    for element in a {
        println!("the value is: {element}");
    }

    for number in (1..4).rev() {
        println!("{number}!");
    }
    println!("Liftoff!");
}
