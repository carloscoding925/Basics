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

class VariableTypes {
    private static int integer = 1; // Integer variable
    private static double decimal = 1.1; // Decimal variable
    private static char character = 'a'; // Character variable
    private static String string = "Hello"; // String variable
    private static boolean bool = true; // Boolean variable
}
