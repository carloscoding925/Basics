package Java.Algorithms;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class GraphsAndTrees {
    public static void main(String[] args) {
        System.out.println("--- This java class will go over common graph and tree algorithms ---");

        treeBfs();

        TreeNode root = new TreeNode(1);
        root.leftNode = new TreeNode(2);
        root.rightNode = new TreeNode(3);
        root.leftNode.leftNode = new TreeNode(4);
        root.leftNode.rightNode = new TreeNode(5);
        int target = 4;
        treeDfs(root, target);

        return;
    }

    static class TreeNode {
        int val;
        TreeNode leftNode;
        TreeNode rightNode;

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode leftNode, TreeNode rightNode) {
            this.val = val;
            this.leftNode = leftNode;
            this.rightNode = rightNode;
        }
    }

    static class GraphNode {
        int val;
        List<GraphNode> neighbors;

        GraphNode(int val) {
            this.val = val;
        }
    }

    /*
     * Breadth First Search for Trees
     * O(n) Time Complexity, O(w) Space Complexity where w is the maximum width
     * Example Provided: Find a target node value using BFS
     */
    private static void treeBfs() {
        TreeNode root = new TreeNode(1);
        root.leftNode = new TreeNode(2);
        root.rightNode = new TreeNode(3);
        root.leftNode.leftNode = new TreeNode(4);
        root.leftNode.rightNode = new TreeNode(5);

        Queue<TreeNode> queue = new LinkedList<>();
        int target = 4;

        queue.offer(root);

        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            System.out.println("Visiting Tree Node: " + node.val);

            if (node.val == target) {
                System.out.println("Found Node with Target: " + node.val);
                return;
            }

            if (node.leftNode != null) {
                queue.offer(node.leftNode);
            }

            if (node.rightNode != null) {
                queue.offer(node.rightNode);
            }
        }

        System.out.println("Target BFS Node Not Found");
        return;
    }

    /*
     * Depth First Search for Trees
     * O(n) Time Complexity, O(h) Space Complexity where h is the height of the tree
     * Example Provided: Find a target node value using DFS
     */
    private static TreeNode treeDfs(TreeNode root, int target) {
        if (root == null) {
            System.out.println("Tree is Empty");
            return null;
        }

        System.out.println("Visiting Tree Node: " + root.val);
        if (root.val == target) {
            System.out.println("Found Node with Target: " + root.val);
            return root;
        }

        TreeNode leftResult = treeDfs(root.leftNode, target);
        if (leftResult != null) {
            return leftResult;
        }

        TreeNode rightResult = treeDfs(root.rightNode, target);
        if (rightResult != null) {
            return rightResult;
        }

        System.out.println("Node with value: " + target + " not found.");
        return null;
    }
}
