using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Random_s : MonoBehaviour
{
    public System.Random ran = new System.Random();
    public float generate_move_str()
    {
        int block = ran.Next(1,55);
        int loc = ran.Next(1,4);
        string move = block.ToString() + "." + loc.ToString();
        return float.Parse(move);
    }
    public Tuple<int,int> generate_move_tup()
    {
        int block = ran.Next(1,55);
        int loc = ran.Next(1,4);
        return Tuple.Create<int,int>(block,loc);
    }



}
