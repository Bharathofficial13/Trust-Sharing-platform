<?php
include("dbconnect.php");
session_start();
extract($_POST);
$id=$_REQUEST['id'];
$name=$_REQUEST['name'];
$qrt=mysql_query("select * from register where id!='$id'");
while($row=mysql_fetch_array($qrt))
{
 $frid=$row['id'];
 $frname=$row['name'];
$int=mysql_query("insert into frlist(id,name,frid,frname,status)values('$id','$name','$frid','$frname','0')");
 $int;
 $int1=mysql_query("insert into frlist(id,name,frid,frname,status)values('$frid','$frname','$id','$name','0')");
 $int1;
}
//header("location:index.php");
?>
<script language="javascript">
	alert("Register Success");
	window.location.href="index.php";
	</script>

	
