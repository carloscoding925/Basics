fn main() {
    let mut x: i32 = 5;
    println!("The value of x is: {x}");
    x = 6;
    println!("The value of x is: {x}");

    const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
    println!("Three Hours in Seconds: {THREE_HOURS_IN_SECONDS}");

    let y: i32 = 5;

    let y: i32 = y + 1;

    {
        let y: i32 = y * 2;
        println!("The value of y in the inner scope is: {y}");
    }

    println!("The value of y is: {y}");

    let spaces: &str = "   ";
    let spaces: usize = spaces.len();

    println!("Spaces length: {spaces}");

    let guess: u32 = "42".parse().expect("Not a Number");
    println!("Guess as a number: {guess}");

    let z: f64 = 2.0;

    let a: f32 = 3.0;

    println!("Floating Points: {}, {}", z, a);

    let t: bool = true;

    let c: char = 'Z';

    println!("Booleans and Characters: {}, {}", t, c);

    let tup: (i32, f64, u8) = (500, 6.4, 1);
    println!("This is a Tuple: {:?}", tup);

    let select_tuple_int: i32 = tup.0;
    println!("The Integer value of the tuple is: {select_tuple_int}");

    let (ti, tf, tu) = tup;
    println!("Or we can destructure a tuple and get the following: {}, {}, {}", ti, tf, tu);

    let empty_unit: () = ();
    println!("Tuples can also be empty which is refered to as a 'unit' : {:?}", empty_unit);

    let arr: [i32; 5] = [1, 2, 3, 4, 5];
    let first_element = arr[0];

    println!("Rust arrays: {:?}", arr);
    println!("First element out of an array: {first_element}");
}
