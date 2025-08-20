package Java.Algorithms;

public class ArraysAndLists {
    public static void main(String args[]) {
        System.out.println("This java class will go over common array and list algorithms");

        oppositeEndTwoPointer();
        slowFastTwoPointer();

        return;
    }

    /*
     * Two Pointer Implementation where two pointers are initialized at opposite ends of an array
     * O(n)
     * Example Provided: Valid, Case Insensitive Palindrome Ignoring Non-Alphanumeric Characters
     */
    private static void oppositeEndTwoPointer() {
        String palindrome = "A man, a plan, a canal: Panama";
        int left = 0;
        int right = palindrome.length() - 1;

        while (left < right) {
            while ((left < right) && (!Character.isLetterOrDigit(palindrome.charAt(left)))) {
                left = left + 1;
            }

            while ((left < right) && (!Character.isLetterOrDigit(palindrome.charAt(right)))) {
                right = right - 1;
            }

            if (Character.toLowerCase(palindrome.charAt(left)) != Character.toLowerCase(palindrome.charAt(right))) {
                System.out.println("Not a valid palindrome");
                return;
            }

            left = left + 1;
            right = right - 1;
        }

        System.out.println("Valid Palindrome");
        return;
    }

    /*
     * Simple Linked List Node Implementation
     * Singly Linked List, meaning nodes only point to the next node, and not the previous
     */
    static class ListNode {
        int val;
        ListNode next;

        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    /*
     * Two Pointer Implementation using slow/fast pointers where one moves one step and the other moves two steps.
     * O(n)
     * Contains both odd and even implementations
     * Example Provided: Find the middle value of a linked list
     */
    private static void slowFastTwoPointer() {
        // Odd Implementation, should land on 2
        ListNode linkedListOne = new ListNode(0, new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4)))));
        ListNode slowOne = linkedListOne;
        ListNode fastOne = linkedListOne;

        while (fastOne != null && fastOne.next != null) {
            fastOne = fastOne.next.next;
            slowOne = slowOne.next;
        }
        System.out.println("Middle value of odd size linked list: " + slowOne.val);

        // Even Implementation, should land in 3
        ListNode linkedListTwo = new ListNode(0, new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5))))));
        ListNode slowTwo = linkedListTwo;
        ListNode fastTwo = linkedListTwo;

        while (fastTwo != null && fastTwo.next != null) {
            fastTwo = fastTwo.next.next;
            slowTwo = slowTwo.next;
        }
        System.out.println("Middle value of even size linked list: " + slowTwo.val);

        return;
    }
}
