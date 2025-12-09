using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CPU : MonoBehaviour
{
    public GameObject trigger;
    public GameObject player;
    void OnMouseDown()
    {
        Random_s scr1 = trigger.GetComponent<Random_s>();
        PlaySequence scr2 = player.GetComponent<PlaySequence>();
        Tuple<int,int> move = scr1.generate_move_tup();
        scr2.SimulateMove(move.Item1,move.Item2);
    }
}
