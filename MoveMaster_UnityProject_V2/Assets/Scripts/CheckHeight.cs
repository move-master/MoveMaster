using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckHeight : MonoBehaviour
{
    public GameObject plane;
    public Camera m_camera;
    private float initialCamY;

    void Start()
    {
        plane = GameObject.Find("Plane");
        m_camera = Camera.main;
        m_camera.clearFlags = CameraClearFlags.Skybox; // trying to fix UI dupe visual bug
        initialCamY = m_camera.transform.position.y;
    }

    void Update()
    {
        if (plane != null)
        {
            float currentY = plane.transform.position.y;
            float diff = currentY - 8.275f;
            int count = 0;

            while (diff > 0)
            {
                diff -= 0.48f;
                count++;
            }

            m_camera.orthographicSize = 4.5f + (0.225f * count);
            float targetY = initialCamY + (0.25f * count);
            m_camera.transform.position = new Vector3(
                m_camera.transform.position.x,
                targetY,
                m_camera.transform.position.z
            );
        }
        else
        {
            Debug.LogError("Plane GameObject not found!");
        }
    }
}
