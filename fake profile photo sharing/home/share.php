<?php
include("dbconnect.php");
session_start();
extract($_POST);
$uid=$_SESSION['uid'];
$fg=mysql_query("select * from register where id='$uid'");
$nm=mysql_fetch_array($fg);
$mn=$nm['name'];
$mid=$_REQUEST['id'];
$max_qry = mysql_query("select max(id) from share");
	$max_row = mysql_fetch_array($max_qry); 
	$id=$max_row['max(id)']+1;
$qrt=mysql_query("insert into share values('$id','$mid','$uid','1')");
if($qrt)
{
header("location:home.php");
}
?>
