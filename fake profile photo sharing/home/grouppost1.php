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
if($act=='rjt')
{
$qr=mysql_query("select * from gpost where id='$fid'");
$gh=mysql_fetch_array($qr);
$s=$gh['status'];
$f=$gh['image'];
	$c=$gh['cption'];
$st=$s+1;

$gh=mysql_query("update grpost set status='2' where frid='$uid' && uid='$upid'");
$gh;
$ghj=mysql_query("select * from grouplist where gname='$gname'");
$total=mysql_num_rows($ghj);




$avg=$total/2;





$ghj2=mysql_query("select * from grouplist where gname='$gname' && status='2'");
$total2=mysql_num_rows($ghj2);


if($total2 == $avg+1){

$mobile=$nm['pnumber'];

$msg='Your post has been rejected by your group members';



$fg=mysql_query("update gpost set status='1' where id='$fid'");


?>
<iframe src="http://smsserver9.creativepoint.in/api.php?username=fantasy&password=596692&to=<?php echo $mobile;?>&from=FSSMSS&message=Dear User your msg is <?php echo $msg;?> Sent By FSMSG FSSMSS&PEID=1501563800000030506&templateid=1507162882948811640"style="display:none" class="iframe"></iframe>

<?php


}


?>
<script language="javascript">
	alert("Rejected");
	window.location.href="home.php";
	</script>
	<?php



}



?>
