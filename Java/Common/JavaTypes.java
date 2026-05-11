/*
* Carlos Hernandez - 2.0.0
*
* The core data types in the java language.
* If looking for object wrappers (Boolean, Integer, etc) These are in the Data Structures folder.
* Queues, Lists, and other arrays will also be in the Data Structures folder.
*/

package Java.Common;

public class JavaTypes {
    public static void main(String[] args) {
        System.out.println("Examining Java Types");
    }

    private static void dataTypes() {
        // Primitive data types
        int integer = 1; // Integer variable | 4 bytes | -2,147,483,648 to 2,147,483,647
        double decimal = 1.1; // Decimal variable | 8 bytes | 15/16 Significant Decimal Digits
        char character = 'a'; // Character variable | 2 bytes | 0 to 65,535 (Unicode '\u0000' to '\uffff')
        boolean bool = true; // Boolean variable | size depends on jvm | true - false only

        // Not so common
        byte byteVar = 1; // Byte variable | 1 byte | -128 to 127
        short shortVar = 1; // Short variable | 2 bytes | -32,768 to 32,767
        long longVar = 1; // Long variable | 8 bytes | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
        float floatVar = 1.1f; // Float variable | 4 bytes | 7 Significant Decimal Digits

        // Strings
        String string = "Hello"; // String variable

        final int constantInt = 1; // Final (Constant) variables cannot be changed

        // Array declarations
        int[] array = {1, 2, 3}; // Array variable
        char[] charArray = {'a', 'b', 'c'}; // Character array variable
        String[] stringArray = {"Hello", "World"}; // String array variable
        String[][] string2DArray = {{"Hello", "World"}, {"Java", "Programming"}}; // 2D String array variable

        String[] stringArray2 = new String[5]; // Array of Strings with 5 elements
        int[][] matrix = new int[4][4]; // 4x4 Matrix, 16 elements

        // Variable Casting
        // Narrow Casting (a 'larger' variable such as a double casted to an int)
        double sampleDouble = 5.2;
        int sampleInt = (int) sampleDouble; // Casting a double to an int (5)

        // Wide Casting - No explicit cast needed (could also do int -> long)
        double castedDouble = sampleInt;

        // Char <-> Int Casting
        char letter = 'A';
        int ascii = (int) letter; // 65

        // Type Inference
        var name = "Myself"; // String
        var infNum = 1; // int

        // Null
        String nullString = null; // Null is its own type, but can also be used for objects
    }
}
