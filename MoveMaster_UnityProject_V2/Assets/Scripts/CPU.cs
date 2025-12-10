using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CPU : MonoBehaviour
{
    public GameObject trigger;
    public GameObject player;
    public GameObject running;
    IEnumerator OnMouseDown()
    {
        Random_s scr1 = trigger.GetComponent<Random_s>();
        PlaySequence scr2 = player.GetComponent<PlaySequence>();
        Tuple<int,int> move;
        int check = 1;
        while (check == 1) {
            move = scr1.generate_move_tup();
            check = scr2.SimulateMove(move.Item1,move.Item2);
        }
        running.SetActive(true);
        yield return new WaitForSeconds(1.5f);
        running.SetActive(false);

    
    }


}
