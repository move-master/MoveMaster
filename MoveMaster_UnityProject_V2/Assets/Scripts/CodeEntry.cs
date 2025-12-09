using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class CodeEntry : MonoBehaviour // Ensure this extends MonoBehaviour
{
    public TMP_InputField inputField; // Assign in Inspector
    public GameObject panel; // UI Panel
    public static string code; // Stores user input
    private CanvasGroup canvasGroup; // Handles UI blocking

    void Start()
    {
        panel = GameObject.Find("PromptPanel");
        canvasGroup = panel.GetComponent<CanvasGroup>();

        if (panel != null)
        {
            panel.SetActive(false); // Ensure the panel starts hidden
        }
    }

    public void SubmitInput()
    {
        if (inputField != null)
        {
            code = inputField.text; // Store user input
            Debug.Log("User entered: " + code);
            PlaySequence.sequence_active = true;
            HideUI(); // Close UI after submission
        }
        else
        {
            Debug.LogError("InputField not assigned! Ensure it's set in the Inspector.");
        }
    }

    public void ShowUI()
    {
        if (panel != null)
        {
            panel.SetActive(true); // Show the UI
            if (canvasGroup != null)
            {
                canvasGroup.interactable = true;
                canvasGroup.blocksRaycasts = true; // Prevent game interaction
            }
        }
    }

    public void HideUI()
    {
        if (panel != null)
        {
            panel.SetActive(false); // Hide UI when done
            Debug.Log(ShowUIScript.temp_camera);
            Switch_Camera.camera_pos = ShowUIScript.temp_camera;
            if (canvasGroup != null)
            {
                canvasGroup.interactable = false;
                canvasGroup.blocksRaycasts = false; // Re-enable game interaction
            }
        }
    }
}
