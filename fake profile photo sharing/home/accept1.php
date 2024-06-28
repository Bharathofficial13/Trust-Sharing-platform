<?php
include("dbconnect.php");
session_start();
extract($_POST);
$uid=$_SESSION['uid'];
$fg=mysql_query("select * from register where id='$uid'");
$nm=mysql_fetch_array($fg);
$mn=$nm['name'];
$fid=$_REQUEST['id'];
$upid=$_REQUEST['upid'];
$act=$_REQUEST['act'];
$gname=$_REQUEST['gname'];

$qr=mysql_query("select * from gpost where id='$fid'");
$gh=mysql_fetch_array($qr);
$s=$gh['status'];
$f=$gh['image'];
	$c=$gh['cption'];
$st=$s+1;

$gh=mysql_query("update grpost set status='1' where frid='$uid' && uid='$upid'");
$gh;
$ghj=mysql_query("select * from grouplist where gname='$gname'");
$total=mysql_num_rows($ghj);




 $avg=$total/2;


$ghj1=mysql_query("select * from grpost where gname='$gname' && status='1'");
 $total1=mysql_num_rows($ghj1);





if($total1 == $avg){

  $mobile=$nm['pnumber'];

$msg='Your post has been accepted by your group members';


?>
<iframe src="http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=<?php echo $mobile;?>&text=Dear customer your msg is <?php echo $msg;?>  Sent By FSMSG FSSMSS" style="display:none" class="iframe"></iframe>




 
   
   <?php
}




?>






<script>
     function myMessage() {
	 
	 alert('accept success');
   window.location.href="home.php";
}


     
   </script>