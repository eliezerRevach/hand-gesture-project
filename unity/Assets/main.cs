using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
using WebSocketSharp.Server;
using System;
using Autohand;
using Unity.VisualScripting;
using System;
using Unity.VisualScripting;
using UnityEngine;

using System.Collections;

    
public class main : MonoBehaviour
{
     public static pos pos;
     public static handCC hand;
     public static String gesture = "rock";
     public class Echo_x : WebSocketBehavior
            {
                protected override void OnMessage(MessageEventArgs  e)
                {
                    float a = Convert.ToInt32(e.Data);
                    // handCC.pos.setX(a);
                }
            }
            public class Echo_y : WebSocketBehavior
            {
                protected override void OnMessage(MessageEventArgs e)
                {
                    float a = Convert.ToInt32(e.Data);
                    pos.sety(a);
                }
            }
            public class Echo_z : WebSocketBehavior
            {
                protected override void OnMessage(MessageEventArgs  e)
                {
                    float a = Convert.ToInt32(e.Data);
                    pos.setz(a);
                }
            }
          
          
          
            public class Echo_rotx : WebSocketBehavior
            {
                protected override void OnMessage(MessageEventArgs e)
                {
                    float a = Convert.ToInt32(e.Data);
                    pos.set_rotaion_X(a);
                }
            }
            public class Echo_roty: WebSocketBehavior
            {
                protected override void OnMessage(MessageEventArgs e)
                {
                    float a = Convert.ToInt32(e.Data);
                    pos.set_rotaion_y(a);
                }
            }
            public class Echo_rotz : WebSocketBehavior
            {
                protected override void OnMessage(MessageEventArgs e)
                {
                    float a = Convert.ToInt32(e.Data);
                    pos.set_rotaion_z(a);
                }
            }



            public class Echo_gesture : WebSocketBehavior
            {
                protected override void OnMessage(MessageEventArgs e)
                {
                    gesture = e.Data;
                }
            }
            

            
    private WebSocketServer wssv;
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("test");
        wssv = new WebSocketServer("ws://127.0.0.1:7891");// create a webscoket to lisen from the live version python file
        wssv.AddWebSocketService<Echo_x>("/x/");
        wssv.AddWebSocketService<Echo_y>("/y/");
        wssv.AddWebSocketService<Echo_z>("/z/");
        
        wssv.AddWebSocketService<Echo_rotx>("/rotx/");
        wssv.AddWebSocketService<Echo_roty>("/roty/");
        wssv.AddWebSocketService<Echo_rotz>("/rotz/");
        
        wssv.AddWebSocketService<Echo_gesture>("/gesture/");
        
        wssv.Start();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
