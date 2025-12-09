using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Running : MonoBehaviour
{

    public GameObject running;
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("working!");
        running = GameObject.Find("Running");
    }

    // Update is called once per frame
    void Update()
    {
        Debug.Log(DestroyOnClick.started);
        if(DestroyOnClick.started == true) {
            running.SetActive(true);
            Debug.Log("active");
        } else {
            running.SetActive(false);
        }
    }
}
