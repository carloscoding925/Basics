// Package refers to the directory where your java file is located in. 
package Java;

// Java files can have multiple classes, but only one public class.
// The public class name must match the file name.
public class JavaCheetSheet {
    // The main method is the entry point of the program.
    public static void main(String[] args) {
        // System.out.println() is used to print to the console.
        System.out.println("Welcome to the Java Cheet Sheet!");
    }
}

class PrintMethods {
    public static void printMethods() {
        System.out.println(""); // Prints with a newline at the end
        System.out.print(""); // No newline at the end
    }
}

class SimpleDataTypes {
    // common
    private static int integer = 1; // Integer variable
    private static double decimal = 1.1; // Decimal variable
    private static char character = 'a'; // Character variable
    private static String string = "Hello"; // String variable
    private static boolean bool = true; // Boolean variable

    // not so common
    private static byte byteVar = 1; // Byte variable
    private static short shortVar = 1; // Short variable
    private static long longVar = 1; // Long variable
    private static float floatVar = 1.1f; // Float variable

    public final static int constantInt = 1; // Final variables cannot be changed
    // public and private refer to how the variable can be accessed.
    // public variables can be accessed from anywhere, private variables can only be accessed from within the same class.

    public static double sampleDouple = 5.2;
    public static int sampleInt = (int) sampleDouple; // Casting a double to an int
}

class JavaOperators {
    public static int int1 = 10;
    public static int int2 = 5;

    public static int result = int1 + int2; // Addition
    public static int result2 = int1 - int2; // Subtraction
    public static int result3 = int1 * int2; // Multiplication
    public static int result4 = int1 / int2; // Division
    public static int result5 = int1 % int2; // Modulus, returns the remainder of the division
    public static int result6 = int1++; // Increment, adds 1 to the variable
    public static int result7 = int1--; // Decrement, subtracts 1 from the variable
}
