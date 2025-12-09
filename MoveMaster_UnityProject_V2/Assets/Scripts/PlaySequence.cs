using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlaySequence : MonoBehaviour
{
    public static bool sequence_active = false;

    // UI References
    public GameObject start;
    public GameObject enter_code;
    public GameObject reset;
    public GameObject camera_button;
    public GameObject status;
    public GameObject running;

    // Movement data
    public List<int> blocks = new List<int>();  // Which block to move
    public List<int> top = new List<int>();     // Where to place (1=left, 2=middle, 3=right)
    private int move_index = 0;

    // Plane & orientation data (from DestroyOnClick)
    public GameObject plane;                // Same Plane object used in DestroyOnClick
    private bool isVertical = false;        // Toggles orientation each layer
    private bool leftOccupied = false;
    private bool middleOccupied = false;
    private bool rightOccupied = false;
    private int placedCount = 0;           // Resets every 3 placements

    // Initial plane setup & positions
    private float planeHeightIncrement = 0.45f;
    private Vector3 initialPlanePosition = new Vector3(50.75f, 8.275f, 50f);
    private Vector3 leftPosition;
    private Vector3 middlePosition;
    private Vector3 rightPosition;

    void Start()
    {
        // Find UI objects
        start         = GameObject.Find("Start Button");
        enter_code    = GameObject.Find("Set Start Button");
        reset         = GameObject.Find("Tower Reset Button");
        camera_button = GameObject.Find("Switch Camera");
        status        = GameObject.Find("Valid_Check");
        running       = GameObject.Find("Running");

        // Find or assign the plane
        if (!plane)
        {
            plane = GameObject.Find("Plane");
        }

        // Initialize positions & orientation
        running.SetActive(false);
        ResetSimulation();
    }

    void Update()
    {
        // If the sequence is triggered, start with a 2-second delay
        if (sequence_active)
        {
            sequence_active = false;
            StartCoroutine(StartSequenceDelayed());
        }
    }

    /// <summary>
    /// Waits 2 seconds, then parses the code and plays the moves.
    /// </summary>
    IEnumerator StartSequenceDelayed()
    {
        Debug.Log("Sequence will start in 2 seconds...");
        yield return new WaitForSeconds(2f);

        // Now parse the user code
        ParseSequence();

        // Play the moves one by one
        yield return StartCoroutine(PlayMovesWithDelay());

        Debug.Log("Sequence Finished.");
    }

    // Parse the user code "X.Y" into blocks (X) and top (Y)
    void ParseSequence()
    {
        // Hide UI
        start.SetActive(false);
        enter_code.SetActive(false);
        reset.SetActive(false);
        camera_button.SetActive(false);
        status.SetActive(false);

        // Clear old data
        blocks.Clear();
        top.Clear();
        move_index = 0;

        // e.g. CodeEntry.code = "1.1 2.2 3.3 4.1 5.2";
        string code = CodeEntry.code; 
        if (string.IsNullOrWhiteSpace(code))
        {
            Debug.LogWarning("No code entered!");
            return;
        }

        // Split by spaces
        string[] nums = code.Split(' ');
        foreach (string n in nums)
        {
            if (float.TryParse(n, out float val))
            {
                int blockID      = Mathf.FloorToInt(val);
                int placementPos = Mathf.RoundToInt((val - blockID) * 10);

                blocks.Add(blockID);
                top.Add(placementPos);
            }
            else
            {
                Debug.LogError($"Invalid input: {n}");
            }
        }

        Debug.Log("Blocks: " + string.Join(", ", blocks));
        Debug.Log("Top: "    + string.Join(", ", top));
    }

    /// <summary>
    /// Executes each move with a wait between them.
    /// After placing each block, sets DestroyOnClick.started = true,
    /// waits 5 seconds, then sets it back to false.
    /// </summary>
    IEnumerator PlayMovesWithDelay()
    {
        while (move_index < blocks.Count)
        {
            int block_from = blocks[move_index];
            int block_to   = top[move_index];

            SimulateMove(block_from, block_to);

            // Toggle started on for 5 seconds after placing
            DestroyOnClick.started = true;
            running.SetActive(true);
            yield return new WaitForSeconds(0.5f);
            running.SetActive(false);
            DestroyOnClick.started = false;

            move_index++;
        }
        yield break;
    }

    // Replicates the exact logic from DestroyOnClick to place blocks
    public void SimulateMove(int blockTag, int topPositionID)
    {
        // 1) Find the block by its tag
        GameObject blockObj = GameObject.FindWithTag(blockTag.ToString());
        if (!blockObj)
        {
            Debug.LogError($"Block with tag {blockTag} not found!");
            return;
        }

        // 2) 'Remove' block from tower
        blockObj.SetActive(false);

        // 3) Calculate placement position, matching DestroyOnClick's logic
        Vector3 targetPos;
        switch (topPositionID)
        {
            case 1: // left
                targetPos = leftPosition;
                if (leftOccupied)
                {
                    Debug.Log("Left position already occupied!");
                    return;
                }
                leftOccupied = true;
                break;
            case 2: // middle
                targetPos = middlePosition;
                if (middleOccupied)
                {
                    Debug.Log("Middle position already occupied!");
                    return;
                }
                middleOccupied = true;
                break;
            case 3: // right
                targetPos = rightPosition;
                if (rightOccupied)
                {
                    Debug.Log("Right position already occupied!");
                    return;
                }
                rightOccupied = true;
                break;
            default:
                Debug.LogError("Invalid top position (must be 1, 2, or 3).");
                return;
        }

        // 4) Place the block
        Debug.Log($"Placing block {blockTag} at {targetPos}");

        // Same orientation logic (horizontal vs vertical)
        blockObj.transform.position = targetPos;
        blockObj.transform.rotation = Quaternion.Euler(0, isVertical ? 90 : 0, 0);
        blockObj.SetActive(true);

        // 5) Increment & check if we need to raise plane
        placedCount++;
        if (placedCount == 3)
        {
            Debug.Log("3 placed! Raising plane.");
            RaisePlane();
        }
    }

    // Copied from DestroyOnClick: toggles orientation, resets occupancy, moves plane up
    void RaisePlane()
    {
        plane.transform.position += new Vector3(0, planeHeightIncrement, 0);
        isVertical = !isVertical;
        SetPlacementPositions();

        leftOccupied   = false;
        middleOccupied = false;
        rightOccupied  = false;
        placedCount    = 0;
    }

    // Same orientation logic from your DestroyOnClick
    void SetPlacementPositions()
    {
        if (isVertical)
        {
            leftPosition   = new Vector3(50.75f, plane.transform.position.y + 0.05f, 49.25f);
            middlePosition = new Vector3(50.75f, plane.transform.position.y + 0.05f, 50f);
            rightPosition  = new Vector3(50.75f, plane.transform.position.y + 0.05f, 50.75f);
        }
        else
        {
            leftPosition   = new Vector3(50f, plane.transform.position.y + 0.05f, 50f);
            middlePosition = new Vector3(50.75f, plane.transform.position.y + 0.05f, 50f);
            rightPosition  = new Vector3(51.5f, plane.transform.position.y + 0.05f, 50f);
        }
    }

    // Resets the plane & orientation back to initial state
    void ResetSimulation()
    {
        // Move plane back to its original pos
        plane.transform.position = initialPlanePosition;

        // Reset orientation
        isVertical     = false;
        placedCount    = 0;
        leftOccupied   = false;
        middleOccupied = false;
        rightOccupied  = false;

        // Recompute positions
        SetPlacementPositions();

        Debug.Log("Simulation Reset");
    }
}
