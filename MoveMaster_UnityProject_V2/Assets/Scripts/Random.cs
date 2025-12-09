using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Random : MonoBehaviour
{
    public System.Random ran = new System.Random();
    float generate_move()
    {
        int block = ran.Next(1,55);
        int loc = ran.Next(1,4);
        string move = block.ToString() + "." + loc.ToString();
        return float.Parse(move);
    }



}
