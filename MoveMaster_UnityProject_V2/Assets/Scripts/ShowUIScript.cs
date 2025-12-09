using UnityEngine;

public class ShowUIScript : MonoBehaviour
{
    public GameObject panel;
    public static int temp_camera = 0;

    void Start()
    {

        if (panel == null)
        {
            Debug.LogError("PromptPanel not found! Check if it's in the scene.");
        }
        else
        {
            panel.SetActive(false); // Ensure the panel starts hidden
        }
    }

    void OnMouseDown()
    {
        if (panel != null)
        {
            PromptPlayer show = panel.GetComponent<PromptPlayer>();
            if (show != null)
            {
                Debug.Log("here!\n");
                show.ShowPrompt();
                temp_camera = Switch_Camera.camera_pos;
                Switch_Camera.camera_pos = 3;
            }
            else
            {
                Debug.LogError("PromptPlayer script not found on PromptPanel!");
            }
        }
    }
}
